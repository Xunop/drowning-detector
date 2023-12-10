import cv2
import torch
import asyncio
import websockets
import threading
from queue import Queue

model = torch.hub.load('/home/xun/Workspace/python/yolov5', 'custom',
                       path='runs/train/exp5/weights/best.pt', source='local')

video_path = "/home/xun/Workspace/python/test_01.mp4"
cap = cv2.VideoCapture(video_path)

CONNECTIONS = set()
frame_queue = Queue()


def init_websocket():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(handler, "0.0.0.0", 8000)
    loop.run_until_complete(start_server)
    loop.run_forever()


async def broadcast(message, conf, frame):
    if message is None or frame is None:
        return
    tasks = [conn.send(message) for conn in CONNECTIONS if conn.open]
    if tasks:
        try:
            await asyncio.gather(*tasks)
        except websockets.exceptions.ConnectionClosed:
            CONNECTIONS.difference_update(
                {conn for conn in CONNECTIONS if not conn.open})


async def process_frame(frame):
    results = model(frame)
    pred = results.xyxy[0]
    message = ""
    conf = 0

    for det in pred:
        label = int(det[5])
        conf = det[4]
        x, y, w, h = map(int, det[:4])
        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

        message = f"Detected object at ({x}, {y}) with confidence {conf}"
        # await broadcast(message, conf)

    return message, conf, frame


async def handler(websocket):
    # Add the client to the set when the connection is established
    CONNECTIONS.add(websocket)

    print(f"New client connected: {websocket.remote_address}")
    async for message in websocket:
        await websocket.send(message)


def video_capture_loop():
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        frame_queue.put(frame)


def video_display_loop():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            message, conf, processed_frame = asyncio.run(process_frame(frame))

            if message != "":
                print(message)
                asyncio.run(broadcast(message, conf, processed_frame))

            if processed_frame is not None:
                cv2.imshow('YOLOv5 Object Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('1'):
                break


websocket_thread = threading.Thread(target=init_websocket)

capture_thread = threading.Thread(target=video_capture_loop)

display_thread = threading.Thread(target=video_display_loop)

websocket_thread.start()
capture_thread.start()
display_thread.start()

websocket_thread.join()
capture_thread.join()
display_thread.join()
