import cv2
import torch
import asyncio
import websockets
import threading

model = torch.hub.load('/home/xun/Workspace/python/yolov5', 'custom', path='runs/train/exp5/weights/best.pt', source='local')

video_path = "/home/xun/Workspace/python/test_01.mp4"
cap = cv2.VideoCapture(video_path)

CONNECTIONS = set()

async def broadcast(message, conf):
    if CONNECTIONS and conf > 0.5:
        await asyncio.gather(*[conn.send(message) for conn in CONNECTIONS])

async def process_frame(frame):
    results = model(frame)
    pred = results.xyxy[0]

    for det in pred:
        label = int(det[5])
        conf = det[4]
        x, y, w, h = map(int, det[:4])
        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

        message = f"Detected object at ({x}, {y}) with confidence {conf}"
        await broadcast(message, conf)

    return frame

async def handler(websocket, path):
    # Add the client to the set when the connection is established
    CONNECTIONS.add(websocket)

    print(f"New client connected: {websocket.remote_address}")

    try:
        while True:
            _, frame = cap.read()
            if frame is None:
                break

            # Perform YOLOv5 inference on the frame
            processed_frame = await process_frame(frame)

            # if process_frame is not None:
                # Convert the processed frame to JPEG and send it to the client
                # _, buffer = cv2.imencode('.jpg', processed_frame)
                # await websocket.send(buffer.tobytes())
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed with {websocket.remote_address}, code: {e.code}, reason: {e.reason}")
    finally:
        # Remove the client from the set when the connection is closed
        CONNECTIONS.remove(websocket)

def video_display_loop():
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        # Perform YOLOv5 inference on the frame
        processed_frame = asyncio.run(process_frame(frame))

        # if process_frame is not None:
            # Convert the processed frame to JPEG and send it to the client
            # _, buffer = cv2.imencode('.jpg', processed_frame)
            # await websocket.send(buffer.tobytes())
        if processed_frame is not None:
            cv2.imshow('YOLOv5 Object Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the video display loop in a separate thread
video_thread = threading.Thread(target=video_display_loop)
video_thread.start()

start_server = websockets.serve(handler, "0.0.0.0", 8000)
asyncio.get_event_loop().run_until_complete(start_server)

try:
    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    pass
finally:
    video_thread.join()
