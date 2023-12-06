import os
from data import multiscalesrdata


class DIV2K(multiscalesrdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2K, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr = super(DIV2K, self)._scan()
        # names_hr = names_hr[self.begin - 1:self.end]

        return names_hr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')

