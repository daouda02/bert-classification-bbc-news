from utils import plot_training_curves

train_losses = [0.7897, 0.0617, 0.0180, 0.0096]
val_losses   = [0.1070, 0.0869, 0.0889, 0.0709]
train_accs   = [73.88, 98.65, 99.61, 99.89]
val_accs     = [97.98, 97.98, 98.43, 98.88]
val_f1s      = [0.9796, 0.9800, 0.9845, 0.9888 ]

plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_f1s,
                     save_path="curves.png")