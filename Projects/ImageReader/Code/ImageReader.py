# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:21:19 2017

@author: Michal
"""

import ImageReaderKlasa

obiekt = ImageReaderKlasa.ImageReader('E:\\projekt_do_XL_Catlyn\\obiekt\\source',
                     'E:\\projekt_do_XL_Catlyn\\obiekt\\results', 'input_file.pdf')

obiekt.splitPDF()
obiekt.convert()
obiekt.reading()