from metabeta.scm.basic import basic_activations
from metabeta.scm.gp import GP
from metabeta.scm.meta import RandomScaleFactory, RandomChoiceFactory

def getActivations():
    activations = basic_activations.copy()
    activations += [GP] * 12
    activations = [RandomScaleFactory(act) for act in activations]
    # ks = [2 ** logUniform(0.1, 4, round=True)
    #       for _ in range(len(activations))]
    # out = [RandomChoiceFactory(activations, int(k)) for k in ks]
    activations += [RandomChoiceFactory(activations)] * 12
    return activations

if __name__ == '__main__':
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    from metabeta.utils import setSeed
    setSeed(42)
    x = torch.arange(start=-10, end=10, step=20/256)

    def gridplot(activations: list[nn.Module], 
                 nrow: int, ncol: int, figsize: tuple[int, int],
                 act_kwargs: dict = {}):

        # simple activations
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axes = axes.flat
        for i,act in enumerate(activations):
            act = act(**act_kwargs)
            y = act(x)
            try:
                axes[i].plot(x,y)
                axes[i].set_title(str(act).split('(')[0], size=10)
            except:
                ...
        fig.tight_layout()

    # simple activations
    gridplot(basic_activations, 7, 4, (7,10))

    # GP-SE activations
    gridplot([GP] * 25, 5, 5, (8,8), dict(gp_type='SE'))

    # GP-Matern activations
    gridplot([GP] * 25, 5, 5, (8,8), dict(gp_type='Matern'))

    # GP-Fract activations
    gridplot([GP] * 25, 5, 5, (8,8), dict(gp_type='Fract'))

    # Random choice activations
    activations = getActivations()
    gridplot(activations[:25], 5, 5, (8,8))


