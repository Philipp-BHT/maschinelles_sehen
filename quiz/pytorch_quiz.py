# Wir wollen von zwei einfachen Funktionen jeweils das Minimum finden.
# 1. f(x) = x^2 + 1
# 2. f(x) = 2*x^2 + 6*x
import torch

# Wofür brauchen wir hier requires_grad? Recherchieren Sie!
# Sorgt dafür, dass die Steigungen des Funktion aufgenommen werden. Jeder Ausdruck mit diesem Tensor wird eine
# differenzierbare Funktion
x = torch.tensor([2.0], requires_grad=True)

# Definieren Sie den SGD Optimizer mit der Lernrate 0.1!
optimizer = torch.optim.SGD([x], lr=0.01)

# Optimization steps
for step in range(100):
    # Gradients auf null setzen
    optimizer.zero_grad()

    # Funktion ausführen
    # y = x**2 + 1
    y = 2*x**2 + 6*x

    # Backpropagation step um die Gradienten zu berechnen
    y.backward()
    # füllt x.grad mit den der Ableitung x'

    # Mit den Gradienten, den Optimizer einen nächsten Schritt machen lassen.
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, f(x) = {y.item():.4f}")

# Finales Resultat bitte copy/paste ins Moodle
# "Minimum at x = ..."
print(f"Minimum at x = {x.item():.4f}")