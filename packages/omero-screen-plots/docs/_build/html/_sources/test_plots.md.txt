# Test Multiple Plots

## First Plot

```{eval-rst}
.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   plt.figure()
   plt.plot([1, 2, 3], [1, 2, 3])
   plt.title("Plot 1")
   plt.show()
```

## Second Plot

```{eval-rst}
.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   plt.figure()
   plt.plot([1, 2, 3], [3, 2, 1])
   plt.title("Plot 2")
   plt.show()
```
