import os
from data import common
from data import multiscalesrdata as srdata


class WHU_RS19(srdata.SRData):
    def __init__(self, args, name='', train=True):
        super(WHU_RS19, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'WHU-RS19', self.name)

        self.dir_hr = os.path.join(self.apath, 'VAL')    ###Airport/Beach/Bridge/Commercial/Desert/Farmland/footballField
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.jpg', '.jpg')
        print(self.dir_hr)
        print(self.dir_lr)
