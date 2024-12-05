import torch
import torch.nn as nn

class PGDAttack:
    def __init__(self, model_target, model_protected=None, eps=0.03, alpha=0.01, lmd=1, iters=40, device="cuda"):
        """
        Initialize the PGD attack class.

        Args:
            model_target: The model to attack.
            model_protected: The model to minimize the attack's impact on (optional).
            eps: Maximum perturbation allowed.
            alpha: Step size for each iteration.
            lmd: Weight for the protected model loss penalty.
            iters: Number of iterations for the attack.
            device: Device to run the attack on ("cuda" or "cpu").
        """
        self.model_target = model_target
        self.model_protected = model_protected
        self.eps = eps
        self.alpha = alpha
        self.lmd = lmd
        self.iters = iters
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        """
        Perform the PGD attack.

        Args:
            images: Input images (torch.Tensor).
            labels: True labels (torch.Tensor).

        Returns:
            Adversarial images (torch.Tensor).
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Store original images
        ori_images = images.clone().detach()

        for _ in range(self.iters):
            images.requires_grad = True

            # Forward pass through the target model
            outputs_target = self.model_target(images)

            # Compute loss for the target model
            loss_target = self.loss_fn(outputs_target, labels)

            # Initialize combined loss with the target model loss
            loss = loss_target

            # Add protected model loss penalty if provided
            if self.model_protected:
                outputs_protected = self.model_protected(images)
                loss_protected = self.loss_fn(outputs_protected, labels)
                loss -= self.lmd * loss_protected

            # Backward pass
            self.model_target.zero_grad()
            if self.model_protected:
                self.model_protected.zero_grad()
            loss.backward()

            # Update adversarial images
            adv_images = images + self.alpha * images.grad.sign()

            # Project perturbations to the epsilon ball and ensure valid pixel values
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        return images
