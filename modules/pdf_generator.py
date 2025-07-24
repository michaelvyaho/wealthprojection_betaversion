import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, "Rapport de Simulation de Patrimoine", 0, 1, 'C')

    def section_title(self, title):
        self.set_font("Arial", 'B', 11)
        self.cell(0, 10, title, 0, 1)

    def section_body(self, body):
        self.set_font("Arial", '', 10)
        self.multi_cell(0, 10, body)