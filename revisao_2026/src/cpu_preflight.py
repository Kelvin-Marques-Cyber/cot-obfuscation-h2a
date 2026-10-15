"""Verifica as importações usadas pela reanálise em um processo novo."""
import os

# O notebook define um backend inline ausente do ambiente virtual isolado.
# O preflight e a reanálise não precisam de janela gráfica.
os.environ["MPLBACKEND"] = "Agg"

import numpy
import pandas
import scipy
import matplotlib
import reportlab
import pypdf
import sklearn
print("Dependências CPU verificadas.")
