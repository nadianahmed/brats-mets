import torch
import numpy as np

from Model.unet_with_attention import AttentionUNet3D
import Model.constants as constants
from Helpers.tensor_helper import crop_or_pad

class ModelUtilities():
    '''
    A class in charge of handling all the model related operations.
    '''
    def __init__(self, type):
        '''
        The constructor for model utilities.

        Parameters:
        - type(String): the type of model.
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if type == "ATTENTION_UNET":
            self.model = AttentionUNet3D(in_channels=constants.ATTENTION_U_NET_NUM_INPUT_CHANNELS, 
                                         out_channels=constants.ATTENTION_U_NET_NUM_OUTPUT_CHANNELS).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=constants.ATTENTION_U_NET_LEARNING_RATE)
        class_weights = torch.tensor(constants.ATTENTION_U_NET_CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def train_one_epoch(self, dataloader):
        '''
        Trains one epoch using the given dataloader.

        Parameters:
        - dataloader(torch.utils.DataLoader): a data loader to train. 
        '''
        self.model.train()
        for batch in dataloader:
            images = batch['image'].to(self.device)
            attn = batch['attention'].to(self.device)
            labels = batch['label'].to(self.device).squeeze(1)

            outputs = self.model(images, attn)  # [N, C, D, H, W]

            # Adjust shapes to match
            outputs = crop_or_pad(outputs, labels.shape[1:])
            labels = crop_or_pad(labels.unsqueeze(1), outputs.shape[2:]).squeeze(1)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"Epoch complete. Loss: {loss.item():.4f}")

    def evaluate_one_epoch(self, dataloader, num_classes=constants.NUM_CLASSES):
        self.model.eval()
        total_loss = 0.0
        total_dice = np.zeros(num_classes)
        total_precision = np.zeros(num_classes)
        total_recall = np.zeros(num_classes)
        total_correct = 0
        total_voxels = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(self.device)
                attn = batch['attention'].to(self.device)
                labels = batch['label'].to(self.device).squeeze(1)  # [N, D, H, W]

                outputs = self.model(images, attn)  # [N, C, D, H, W]
                outputs = crop_or_pad(outputs, labels.shape[1:])
                labels = crop_or_pad(labels.unsqueeze(1), outputs.shape[2:]).squeeze(1)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)  # [N, D, H, W]

                # Compute metrics
                batch_dice = self.evaluate_dice_score(preds, labels, num_classes)
                batch_precision = self.evaluate_precision_score(preds, labels, num_classes)
                batch_recall = self.evaluate_recall_score(preds, labels, num_classes)
                total_dice += np.array(batch_dice)
                total_precision += np.array(batch_precision)
                total_recall += np.array(batch_recall)
                total_correct += (preds == labels).sum().item()
                total_voxels += torch.numel(labels)
                num_batches += 1

        average_loss = total_loss / num_batches
        average_dice = total_dice / num_batches
        average_precision = total_precision / num_batches
        average_recall = total_recall / num_batches
        average_accuracy = total_correct / total_voxels

        print(f"Evaluation complete.")
        print(f"  Average Loss: {average_loss:.4f}")
        print(f"  Average Accuracy: {average_accuracy:.4f}")
        for cls in range(num_classes):
            print(f"  Class {cls}: Dice = {average_dice[cls]:.4f}, Precision = {average_precision[cls]:.4f}, Recall = {average_recall[cls]:.4f}")

        return average_loss, average_accuracy, average_dice, average_precision, average_recall
    
    # --- Evaluation functions ---

    def evaluate_dice_score(self, pred, target, num_classes):
        '''
        Calculates the dice score from the predictions.

        Parameters:
        - pred(Int): the predicted class.
        - target(Int): the target class.
        - num_class(Int): number of classes.

        Returns:
        - Float: the calculated the dice score.
        '''
        dice_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            dice = (2. * intersection + constants.SMOOTHNESS_FACTOR) / (pred_cls.sum() + target_cls.sum() + constants.SMOOTHNESS_FACTOR)
            dice_per_class.append(dice.item())
        return dice_per_class

    def evaluate_precision_score(self, pred, target, num_classes):
        '''
        Calculates the precision score from the predictions.

        Parameters:
        - pred(Int): the predicted class.
        - target(Int): the target class.
        - num_class(Int): number of classes.

        Returns:
        - Float: the calculated the prediction score.
        '''
        precision_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            tp = (pred_cls * target_cls).sum()
            fp = (pred_cls * (1 - target_cls)).sum()
            precision = (tp + constants.SMOOTHNESS_FACTOR) / (tp + fp + constants.SMOOTHNESS_FACTOR)
            precision_per_class.append(precision.item())
        return precision_per_class

    def evaluate_recall_score(self, pred, target, num_classes):
        '''
        Calculates the recall score from the predictions.

        Parameters:
        - pred(Int): the predicted class.
        - target(Int): the target class.
        - num_class(Int): number of classes.

        Returns:
        - Float: the calculated the recall score.
        '''
        recall_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            tp = (pred_cls * target_cls).sum()
            fn = ((1 - pred_cls) * target_cls).sum()
            recall = (tp + constants.SMOOTHNESS_FACTOR) / (tp + fn + constants.SMOOTHNESS_FACTOR)
            recall_per_class.append(recall.item())
        return recall_per_class