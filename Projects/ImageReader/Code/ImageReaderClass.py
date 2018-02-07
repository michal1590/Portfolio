# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:17:54 2017

@author: Michal

Objektowa wersja
"""
import os

class ImageReader:
    
 
    def __init__(self,sourceURL, resultsURL, sourceName, tmpURL = None):
        self.sourceURL = sourceURL
        self.resultsURL = resultsURL
        self.sourceName = sourceName
        if tmpURL == None:
            os.mkdir(self.resultsURL + '\\temporary')
            self.tmpURL = self.resultsURL + '\\temporary'
        else:
            self.tmpURL = tmpURL                                 

    
    
    def getName(self):
        return self.sourceName
    
    
    def splitPDF(self, startPage=1, endPage=3):
        
        from PyPDF2 import PdfFileReader, PdfFileWriter
        
        self.startPage = startPage
        self.endPage = endPage+1
        
        os.chdir(self.sourceURL)
        inputPDF = PdfFileReader(self.sourceName, 'wb')

        os.chdir(self.tmpURL)
        for page in range(self.startPage, self.endPage):
            outputPDF = PdfFileWriter()
            outputPDF.addPage(inputPDF.getPage(page))
            with open ('tmpPDF%s.pdf' %page, 'wb') as file:
                outputPDF.write(file)    
        
                  
                
        
    def convert(self):
        
#        from PythonMagick import Image, CompositeOperator.SrcOverCompositeOp
        import PythonMagick
        import cv2
        
        os.chdir(self.tmpURL)
                
        for page in range(self.startPage, self.endPage):
            img = PythonMagick.Image()
            img.density('300')
            img.read('tmpPDF%s.pdf' %page) 
            size = "%sx%s" % (img.columns(), img.rows())
            
            bg_colour = "#ffffff" # specyfing background color is manadatory 
            output_img = PythonMagick.Image(size, bg_colour) 
            output_img.composite(img, 0, 0, 
                                 PythonMagick.CompositeOperator.SrcOverCompositeOp)
            output_img_url = self.tmpURL + ('\\img%s.PNG' %page)
            output_img.write(output_img_url) 
        
            #binarization
            img = cv2.imread(output_img_url,cv2.IMREAD_GRAYSCALE)
            x,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            cv2.imwrite(output_img_url,img)
        
        
    
    
    def reading(self):
#        from pytesseract import pytesseract  
        import pytesseract
        os.chdir(self.resultsURL)
#        print(output_img_url)
        for page in range(self.startPage, self.endPage):
            pytesseract.pytesseract.run_tesseract(self.tmpURL+('\\img%s.PNG' %page), 
                                      'text_page%s' %page, lang='eng',boxes=False,
                                      config="hocr")
            
        
        

#
obiekt = ImageReader('E:\\portfolio\\Portfolio\\ImageReader\\Other\\Source',
                     'E:\\portfolio\\Portfolio\\ImageReader\\Other\\Results', 'input_file.pdf')

#E:\portfolio\Portfolio\ImageReader\Other
obiekt.splitPDF(1,2)
obiekt.convert()
obiekt.reading()
