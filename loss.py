import torch
import torch.nn as nn

import cv2
import numpy as np


class dice_bce_loss(nn.Module):

    def soft_dice_coeff(self, y_pred, y_true):
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        smooth = 1e-20
        i = torch.sum(y_pred)
        j = torch.sum(y_true)
        intersection = torch.sum(y_true * y_pred)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # IoU loss
        return 1 - score.mean()

    def __call__(self, y_pred, y_true, use_half_training):
        assert torch.isnan(y_pred).sum() == 0 and torch.isinf(y_pred).sum() == 0, ('y_pred is nan or ifinit', y_pred)
        if use_half_training:
            a = self.bce_loss(y_pred.type(torch.FloatTensor), y_true.type(torch.FloatTensor))
        else:
            a = self.bce_loss(y_pred.type(torch.FloatTensor), y_true.type(torch.FloatTensor))

        assert torch.isnan(a).sum() == 0 and torch.isinf(a).sum() == 0, ('bce_loss is nan or ifinit', a)
        b = self.soft_dice_coeff(y_pred, y_true)

        return 0.7*a+0.3*b


class SegmentationMetric(object):
    def __init__(self, numClass, ignore_labels=None):
        self.numClass = numClass
        self.ignore_labels = ignore_labels
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def Kappa(self):
        p0 = self.pixelAccuracy()
        pc = 0
        for i in range(self.confusionMatrix.shape[0]):
            pc = pc + self.confusionMatrix[i].sum() * self.confusionMatrix[:, i].sum()
        pc = pc / self.confusionMatrix.sum()**2
        kappa = (p0 - pc) / (1 - pc)
        return kappa

    def classPixelPrecision(self):
        classPrecision = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)  # [H,W]->[0,1]
        return classPrecision

    def classPixelRecall(self):
        classRecall = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)  # [H,W]->[0,1]
        return classRecall

    def classPixelF1(self):
        """
        F1 = 2*P*R/(R+P)
        """
        classF1 = 2*self.classPixelPrecision()*self.classPixelRecall()/(self.classPixelPrecision()+self.classPixelRecall())
        return classF1

    def meanPixelPrecision(self):
        classRecall = self.classPixelRecall()
        meanCorrect = classRecall[classRecall < float('inf')].mean()
        return meanCorrect

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)
        if self.ignore_labels != None:
            union[self.ignore_labels] = 0
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)  # same_shape bool_type
        if self.ignore_labels != None:
            for IgLabel in self.ignore_labels:
                mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label.type(torch.IntTensor), minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusionMatrix, axis=1) / torch.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (torch.sum(self.confusionMatrix, axis=1) +
                                              torch.sum(self.confusionMatrix, axis=0) - torch.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)

    def evalus(self, imgPredict, imgLabel):
        hist = self.addBatch(imgPredict, imgLabel)
        Accuracy = self.pixelAccuracy()
        kappa = self.Kappa()
        Precision = self.classPixelPrecision()
        Recall = self.classPixelRecall()
        F1 = self.classPixelF1()
        MPA = self.meanPixelPrecision()
        IoU = self.IntersectionOverUnion()
        mIoU = self.meanIntersectionOverUnion()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        evalus_res = 'Precision : {}, Recall : {}, F1 : {}, IoU : {}, Accuracy : {}, kappa : {}, MPA : {},  mIoU : {}, FWIoU : {}'.format(
                                Precision.numpy(), Recall.numpy(), F1.numpy(), IoU.numpy(),
                                Accuracy.numpy(), kappa.numpy(), MPA.numpy(), mIoU.numpy(), FWIoU.numpy())
        self.reset()
        return evalus_res



if __name__ == '__main__':
    imgPredict = torch.tensor([[[0,1,2],[2,1,1]],
                                [[0,1,2],[2,0,1]],
                                [[0,1,2],[2,2,1]]
                                                    ]).long()
    imgLabel = torch.tensor([[[0,1,0],[1,1,2]],
                            [[0,1,1],[1,1,0]],
                            [[0,1,2],[1,0,2]]
                                                ]).long()
    ignore_labels = None
    metric = SegmentationMetric(3, ignore_labels)
    metric.evalus(imgPredict, imgLabel)

