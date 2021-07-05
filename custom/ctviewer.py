from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import nibabel
import pathlib


COLORTABLE=[]
for i in range(256):
    def crop(x):
        return min(max(0, int(x)), 255)

    # matlab jet colortable https://stackoverflow.com/questions/58666021/how-can-i-apply-a-colortable-to-my-grayscale-8-bit-image-and-convert-it-correctl
    r = -4.0 * abs(i - 255.0 * 3.0 / 4) + 255.0 * 3.0/ 2
    g = -4.0 * abs(i - 255.0 * 2.0 / 4) + 255.0 * 3.0 / 2
    b = -4.0 * abs(i - 255.0 * 1.0 / 4) + 255.0 * 3.0 / 2
    COLORTABLE.append(QtGui.qRgb(crop(r), crop(g), crop(b)))

dmin = -1024
dmax = 4000


class CtImage(QtWidgets.QVBoxLayout):
    def __init__(self, file:str):
        super().__init__()
        self.file = file
        self.not_opened = True
        f = pathlib.Path(file)
        if not f.exists():
            print(f'open file{file} failed')
            return
        try:
            self.img: nibabel.Nifti1Image = nibabel.load(file)
        except Exception as ex:
            print('ex', ex)
            return
        title = f.name.replace('.nii.gz', '')
        label = QtWidgets.QLabel(title)
        print(f'title = {title} file = {file}', title)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("color:red;font:32px")
        self.addWidget(label)

        image = QtWidgets.QLabel()
        image.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(image, stretch=1)

        self.affine = self.img.affine
        img_data: np.array = self.img.get_fdata()
        max_data_value = np.max(img_data)
        print(f'max_data for {file}', max_data_value)
        self.img_data = (img_data - dmin) * 255 / (dmax - dmin)
        w, h = self.img_data.shape[1], self.img_data.shape[0]
        print(self.img_data.shape)
        qimage_data = np.require(self.img_data[:,:,45], np.uint8, requirements='C')
        qimage = QtGui.QImage(qimage_data, w, h, QtGui.QImage.Format_Indexed8)
        qimage.setColorTable(COLORTABLE)

        self.width = img_data.shape[1]
        # self.height = img_data.shape[0]
        image.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.zrange = self.affine[2][3], self.affine[2][3] + self.affine[2][2] * (self.img_data.shape[2] - 1)
        print('setPixmap done')
        self.image = image
        self.not_opened = False

    def set_z(self, z):
        scale = self.affine[2][2]
        offset = self.affine[2][3]
        # z = scale * index + offset => index = (z - offset) / scale
        index = (int)((z - offset) / scale)
        if index < 0:
            index = 0
        elif index > self.img.shape[2]:
            index = self.img.shape[2] - 1
        qimage_data = np.require(self.img_data[:,:,index], np.uint8, requirements='C')
        qimage = QtGui.QImage(qimage_data, self.img_data.shape[1], self.img_data.shape[0], QtGui.QImage.Format_Indexed8)
        qimage.setColorTable(COLORTABLE)
        self.image.setPixmap(QtGui.QPixmap.fromImage(qimage))


class ScrollArea(QtWidgets.QScrollArea):
    openImageRequested = QtCore.pyqtSignal(str)

    def __init__(self):

        super().__init__()
        self.setAcceptDrops(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollWidget = QtWidgets.QWidget()
        self.setWidget(scrollWidget)
        self.setWidgetResizable(True)
        viewerBox = QtWidgets.QHBoxLayout()
        self.viewerBox = viewerBox
        scrollWidget.setLayout(viewerBox)

    def add(self, ct_viewer):
        self.viewerBox.addLayout(ct_viewer)

    def set_z(self, z):
        for ch in self.viewerBox.children():
            print(ch)
            if isinstance(ch, CtImage):
                ch.set_z(z)

    def dragEnterEvent(self, e):
        mime = e.mimeData()
        print(mime)

        if mime.hasUrls():
            print(mime.urls())
            for url in mime.urls():
                print(url.toLocalFile())
        if mime.hasFormat('text/plain') or mime.hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        print(e.mimeData())
        mime = e.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                folder = url.toLocalFile()
                folder = folder.replace('/', '\\')
                print('drop folder', folder)
                self.openImageRequested.emit(folder)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT(PET) Viewer")
        self.box = box = QtWidgets.QVBoxLayout()
        self.setLayout(box)
        self.setGeometry(300, 300, 800, 600)
        # box.addWidget(QtWidgets.QLabel("XXX"))
        # box.addWidget(QtWidgets.QLabel("YYYY"))
        # self.show()
        # return
        scroll = ScrollArea()
        def new_image(f):
            print('new image', f)
            self.addCt(f)
        scroll.openImageRequested.connect(new_image)
        box.addWidget(scroll, stretch=1)
        # scroll.setFrameShape(frame)
        self.viewerBox = scroll

        ct_file = "CT_crop.nii.gz"
        self.zrange = None
        c = QtWidgets.QLabel("Control")
        c.setStyleSheet("color: red;font: bold 32px")
        box.addWidget(c)
        self.init_controller()
        self.show()
        self.addCt(ct_file)

    def update_slider(self):
        self.slider.setRange(*self.zrange)

    def init_controller(self):
        hbox = QtWidgets.QHBoxLayout()
        self.box.addLayout(hbox)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        hbox.addWidget(self.slider)
        self.slider.move(30, 30)
        self.slider.setSingleStep(1)
        def slider_change(*v):
            print('slider changed', v)
            self.viewerBox.set_z(v)
        self.slider.valueChanged.connect(slider_change)

    def addCt(self, file):
        ct = CtImage(file)
        if ct.not_opened:
            return
        if self.zrange:
            zmin = min(self.zrange[0], ct.zrange[0])
            zmax = max(self.zrange[1], ct.zrange[1])
            self.zrange = zmin, zmax
        else:
            self.zrange = tuple(ct.zrange)
        self.viewerBox.add(ct)
        self.update_slider()


if __name__ == '__main__':
    import sys
    sys._excepthook = sys.excepthook
    import traceback
    def exception_hook(exctype, value, tracebackobj):
        traceback.print_exception(exctype, value, tracebackobj)
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())
