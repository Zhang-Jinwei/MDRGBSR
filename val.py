from option import args
import torch
import utility
import data
import model
import loss
import os
from trainer import Trainer

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        args.test_only = True
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        # while not t.terminate():
        for i in range(0, 600):
            if i > 0:
                model.get_model().load_state_dict(
                    torch.load(os.path.join('./experiment/blindsr_x4_bicubic_iso', 'model', 'model_add_{}.pt'.format(i+1))),  ###
                    ###
                    strict=False
                )
            with open("test_add.txt", "a") as f:
                f.write('epoch:\t' + str(i+1) + '\t\n')
            t.terminate()
            # t.test()

        checkpoint.done()
