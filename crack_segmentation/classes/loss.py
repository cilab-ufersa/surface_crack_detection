import sys


class Loss:
    def __init__(self, args):
        self.args = args

    def define_Loss(self):
        sys.path.append(self.args["main"])

        # choose loss
        if self.args['loss'] == 'Focal_Loss':
            from subroutines.loss_metric import Focal_Loss
            loss = Focal_Loss(
                alpha=self.args['focal_loss_a'], gamma=self.args['focal_loss_g'])

        elif self.args['loss'] == 'WCE':
            from subroutines.loss_metric import Weighted_Cross_Entropy
            loss = Weighted_Cross_Entropy(beta=self.args['WCE_beta'])

        elif self.args['loss'] == 'F1_score_Loss':
            from subroutines.loss_metric import F1_score_Loss
            loss = F1_score_Loss

        elif self.args['loss'] == 'F1_score_Loss_dil':
            from subroutines.loss_metric import F1_score_Loss_dil
            loss = F1_score_Loss_dil

        elif self.args['loss'] == 'Binary_Crossentropy':
            import tensorflow as tf
            loss = tf.keras.losses.BinaryCrossentropy()

        return loss
