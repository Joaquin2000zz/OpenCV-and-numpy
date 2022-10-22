"""
module which contains QR's class
"""
import base64
import cv2
import io
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
import qrcode


class QR:
    """
    class that creates and decodes qr's codes
    """

    def create(self, data=None, path=None, PIL=False, for_pdf=False,
               fill_color='black', back_color='white',
               version=1, box_size=10, border=5):
        """
        create and saves qr's codes in a given path or makes a PIL object
        data: QR's content
        path: QR's path in which gonna be saved
        PIL: flag which determines whether it's gonna
             return an PIL object or not
        for_pdf: flag which determines wheter it's gonna return
                an PIL image streamed in Bytes for be ready to
                use in reportlab instance (to insert the qr in a pdf)
        fill_color: QR's color
        back_color: QR's background color
        version: QR's version
        box_size: QR's size
        border: QR's border size
        Return: - if a path is given, returns True in success
                - if PIL is True, returns a PIL object with the image
                - otherwise, if it fails, returns False
        """

        if not data or (not path and not PIL and not for_pdf):
            return False

        # Creating an instance of QRCode class
        qr = qrcode.QRCode(version=version,
                           box_size=box_size,
                           border=border)

        # Adding data to the instance 'qr'
        qr.add_data(data)

        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color,
                            back_color=back_color)

        if for_pdf:
            # just use reportlab.lib.utils.ImageReader() before
            # passing to the reportlab.pdfgen.canvas.Canvas().drawImage()
            # instance method << Canvas() instance >> and wuoalah :)
            stream = io.BytesIO()
            img.save(stream=stream, format='png')
            return stream

        if PIL:
            return img

        if path[-4:] != '.png':
            path += '.png'
        img.save(path)

        return True

    def decode(self, img=None, path=None):
        """
        decodes an image and if it finds a QR
        extract its information of its barcode
        Return: list with the detections. otherwise, returns False
        """

        if path:
            if path[-4:] != '.png':
                path += '.png'
            img = cv2.imread(path)

        decoded = decode(img)

        if not decoded:
            return False

        retList = []
        # creating list of dictionaries which contains the decoded message
        # and the bounding box of the QR detected
        for barcode in decoded:
            retList.append({
                'content': barcode.data.decode('utf-8'),
                'polygon_box': np.array([barcode.polygon],
                                         np.int32).reshape((-1, 1, 2)),
                'bounding_box': barcode.rect
            })

        # if decodes an barcode, we gonna return both, their content and position
        return retList

    def start(self):
        """
        open the camera and starts to capture video
        """

        video = cv2.VideoCapture(0)

        video.set(3, 640)
        video.set(4, 480)

        while True:
            flag, frame = video.read()

            if flag:
                decoded = self.decode(frame)

                if decoded:
                    for content in decoded:
                        cv2.polylines(img=frame, isClosed=True,
                                      pts=[content['polygon_box']],
                                      color=(0, 0, 255),
                                      thickness=2)
                        l, t, w, h = content['bounding_box']
                        print(content['bounding_box'])

                        # Finds space required by the text so that
                        # we can put a background with that amount of width.
                        (w, h), _ = cv2.getTextSize(content['content'],
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.9, 1)
                        cv2.rectangle(frame, (l, t - 40),
                                      (l + w, t), (0, 0, 255), -1)
                        cv2.putText(img=frame, text=content['content'],
                                    org=(l, t - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.9,
                                    thickness=1,
                                    color=(0, 0, 0))
            cv2.imshow('Scanner', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def read_bytes_array(self, bytes):
        """
        read bytes array and decode their content
        if it contains a qr. Otherwise, returns False  
        """
        try:
            imgdata = base64.b64decode(bytes.split(',')[1])
            img = Image.open(io.BytesIO(imgdata))
            opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

            return self.decode(img=opencv_img)
        except:
            return False

if __name__ == '__main__':
    qr = QR()
    qr.start()
