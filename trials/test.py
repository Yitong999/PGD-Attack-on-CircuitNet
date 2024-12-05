import torch
from torch.utils.data import DataLoader
from pgd_attack import PGDAttack
from datasets.build_dataset import build_dataset
from models.build_model import build_model
from utils.configs import Parser

def test():
    # Parse arguments
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    # Load dataset
    print("===> Loading datasets for testing")
    arg_dict['test_mode'] = True
    dataset = build_dataset(arg_dict)
    dataloader = DataLoader(dataset, batch_size=arg_dict['batch_size'], shuffle=False)

    # Load models
    print("===> Building models")
    model_target = build_model(arg_dict)  # Model to attack
    model_protected = build_model(arg_dict)  # Model to preserve

    if not arg_dict['cpu']:
        model_target = model_target.cuda()
        model_protected = model_protected.cuda()

    # Load pre-trained weights
    model_target.load_state_dict(torch.load(arg_dict['model_target_path'])['state_dict'])
    model_protected.load_state_dict(torch.load(arg_dict['model_protected_path'])['state_dict'])

    # Initialize PGD attack
    pgd_attack = PGDAttack(
        model_target=model_target,
        model_protected=model_protected,
        eps=0.03,            # Maximum perturbation
        alpha=0.01,          # Step size
        lmd=1,               # Lambda for the penalty
        iters=40,            # Number of iterations
        device='cuda' if not arg_dict['cpu'] else 'cpu'
    )

    model_target.eval()
    model_protected.eval()

    # Evaluation
    correct_target_adv = 0
    correct_protected_adv = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            if not arg_dict['cpu']:
                images, labels = images.cuda(), labels.cuda()

            # Generate adversarial examples
            adv_images = pgd_attack.attack(images, labels)

            # Predictions on adversarial examples
            pred_target_adv = torch.argmax(model_target(adv_images), dim=1)
            pred_protected_adv = torch.argmax(model_protected(adv_images), dim=1)

            # Calculate accuracy
            correct_target_adv += (pred_target_adv == labels).sum().item()
            correct_protected_adv += (pred_protected_adv == labels).sum().item()
            total += labels.size(0)

    print("===> Evaluation Results:")
    print(f"Model Target Accuracy on Adversarial Examples: {correct_target_adv / total * 100:.2f}%")
    print(f"Model Protected Accuracy on Adversarial Examples: {correct_protected_adv / total * 100:.2f}%")

if __name__ == "__main__":
    test()
