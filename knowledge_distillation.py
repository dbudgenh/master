import torch.nn.functional as F
import torch.nn as nn

def knowledge_distillation_loss(student_output,teacher_output,labels,alpha=0.95,T=3.5):
    """Compute the KD-loss between student_output, teach_output and the correct labels. 

    Args:
        student_output (Tensor): Logits (unnormalized output) of the student model
        teacher_output (Tensor): Logits (unnormalized output) of the teacher model
        labels (Tensor): Integer between [0,C]
        alpha (float, optional): alpha value, as described in the paper. Defaults to 0.95.
        T (float, optional): temperature value for the softmax, as described in the paper. Defaults to 3.5.
    """
    loss = nn.KLDivLoss()(F.log_softmax(student_output/T,dim=1),
                        F.softmax(teacher_output/T,dim=1)) * (alpha * T * T) + F.cross_entropy(student_output,labels) * (1. - alpha)
    return loss
