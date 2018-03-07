from picamera import PiCamera
from http.server import HTTPServer, SimpleHTTPRequestHandler
from signal import signal, SIGINT
import pickle
import numpy as np
import sys
import io

cam = PiCamera()

cam_ver = {
        'RP_ov5647': 1,
        'RP_imx219': 2,
}[cam.exif_tags['IFD0.Model']]
print("Detected RPi Cam Ver. ", cam_ver)

raw_offset = {
        1: 6404096,
        2: 10270208,
}[cam_ver]

cam_reshape = {
        1: (1952, 3264),
        2: (2480, 4128),
}[cam_ver]
cam_crop = {
        1: (1944, 3240),
        2: (2464, 4100),
}[cam_ver]

def close(signal, frame):
    cam.close()
    print("[INFO] Closed camera")
    sys.exit(0)

class CameraRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()

            self.wfile.write(bytes('Hello World\n', 'utf-8'))
        elif self.path.startswith('/image'):
            try:
                args = self.path.split('?')[1].split('&')
                reqs = dict()
                for arg in args:
                    key, val = arg.split('=')
                    if val.isnumeric():
                        reqs[key] = int(val)
                    else:
                        reqs[key] = val

                if ('x' in reqs) and ('y' in reqs):
                    cam.resolution = (reqs['x'], reqs['y'])

                capture_raw = 'raw' in reqs
                print("capturing raw: {}".format(capture_raw))
                
                img = io.BytesIO()
                cam.capture(img, format='jpeg', bayer=capture_raw)

                if capture_raw:
                    data = img.getvalue()[-raw_offset:]
                    print(data[:4])
                    assert data[:4] == bytes('BRCM', 'ascii')
                    data = data[32768:]
                    arr = np.fromstring(data, dtype=np.uint8)
                    arr = arr.reshape((cam_reshape[0], cam_reshape[1]))
                    arr = arr[:cam_crop[0], :cam_crop[1]]


                    # Convert to 16-bit
                    arr = arr.astype(np.uint16) << 2
                    for byte in range(4):
                        arr[:, byte::5] |= ((arr[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
                    arr = np.delete(arr, np.s_[4::5], 1)

                    # Split into RGB from BGGR
                    rgb = np.zeros(arr.shape + (3,), dtype=arr.dtype)
                    rgb[1::2, 0::2, 0] = arr[1::2, 0::2] # Red
                    rgb[0::2, 0::2, 1] = arr[0::2, 0::2] # Green
                    rgb[1::2, 1::2, 1] = arr[1::2, 1::2] # Green
                    rgb[0::2, 1::2, 2] = arr[0::2, 1::2] # Blue

                    print(rgb.shape)
                    print("Processing")
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.send_header('X-Image-Height', rgb.shape[0])
                    self.send_header('X-Image-Width', rgb.shape[1])
                    self.send_header('X-Image-Channels', rgb.shape[2])
                    self.end_headers()

                    print("Processing 2")
                    """
                    imgBuffer = io.BytesIO()
                    print("Datatype:",rgb.dtype)
                    m_img = Image.fromarray(rgb[:,:,0])
                    m_img.save(imgBuffer, 'tiff')
                    imgBuffer.seek(0)
                    """
                    red_channel = rgb[1::2,::2,0]
                    self.wfile.write(pickle.dumps(red_channel))


                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.end_headers()
                    
                    img.seek(0)
                    self.wfile.write(img.read())

            except Exception as e:
                raise e
                print(e)
                self.send_error(500)
                self.end_headers()


signal(SIGINT, close)
address = ('', 8000)
httpd = HTTPServer(address, CameraRequestHandler)
httpd.serve_forever()
