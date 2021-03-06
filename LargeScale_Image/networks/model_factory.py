import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, trainer, taskcla):
        if trainer == 'hat':
            import networks.alexnet_hat as alex
            return alex.alexnet(taskcla, pretrained=False)
        elif 'coscl' in trainer:
            import networks.alexnet_coscl as alex
            return alex.alexnet(taskcla, pretrained=False)
        else:
            import networks.alexnet as alex
            return alex.alexnet(taskcla, pretrained=False)