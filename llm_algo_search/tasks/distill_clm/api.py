from torch import nn

class API(nn.Module):
    def forward(self, student_logits, target_logits, labels):
        """
        student_output.shape (batch, sequence, model)
        target_output.shape (batch, sequence, model)
        labels.shape (batch, sequence)
        """
        pass
