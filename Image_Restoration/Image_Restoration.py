from tkinter import*
from PIL import Image, ImageTk

import numpy as np
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq, rfft2

import matplotlib
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

import cv2
import math
import __future__
import sys
import time
import getopt

class MainWindow():
    # Конструктор класса MainWindow
    def __init__(self):
        # ------------------------------------------------ Переменные класса-------------------------------------------------
        self.image = None
        self.blur_type = IntVar()
        self.angle = None
        self.radius = None
        self.sharpness = None
        self.noise = None
        self.cv_img = None
        self.furier_img = None
        self.psf_pic = None
        self.result_image = None
        self.start_image = None
        self.path = None
        self.res = None
        self.spectrum_res = None
        # -------------------------------------------------- Тип смазывания -------------------------------------------------
        self.label_frame = LabelFrame(root, text='Types of Blur', bg='#fffff0', bd=1, fg='#696969', font='Courier 16', height=100, width=300)
        self.label_frame.place(relx=0.025, rely=0.025)

        self. rb1 = Radiobutton(self.label_frame, text='Out of focus', background='#fffff0', fg='#696969', font='Courier 14', variable=self.blur_type, value=1, command=lambda par=1:self.process_image(par)) 
        self.rb1.pack(expand=1, anchor=W)
        self.rb2 = Radiobutton(self.label_frame, text='Motion', background='#fffff0', fg='#696969', font='Courier 14', variable=self.blur_type, value=0, command=lambda par=0:self.process_image(par))
        self.rb2.pack(expand=1, anchor=W)

        # ------------------------------------------------ Параметры ------------------------------------------------
        self.par_frame = LabelFrame(root, text='Settings', bg='#fffff0', bd=1, fg='#696969', font='Courier 16', height=150, width=300)
        self.par_frame.place(relx=0.025, rely=0.2)

        self.angle_label = Label(self.par_frame, text='Angle:', bg='#fffff0', fg='#696969', font='Courier 12')
        self.angle_label.pack(expand=1, anchor=W)

        self.angle_scrollbar = Scale(self.par_frame, bg='#fffff0', cursor='hand1', from_=0, to=180, orient=HORIZONTAL, showvalue=1, relief=FLAT, length=172, command=self.process_image)
        self.angle_scrollbar.pack(expand=1, anchor=W)
        
        self.radius_label = Label(self.par_frame, text='Radius:', bg='#fffff0', fg='#696969', font='Courier 12')
        self.radius_label.pack(expand=1, anchor=W)

        self.radius_scrollbar = Scale(self.par_frame, bg='#fffff0', cursor='hand1', from_=0, to=50, orient=HORIZONTAL, showvalue=1, variable=self.radius, relief=FLAT, length=172, command=self.process_image)
        self.radius_scrollbar.pack(expand=1, anchor=W)
       
        self.sharpness_label = Label(self.par_frame, text='Sharpness:', bg='#fffff0', fg='#696969', font='Courier 12')
        self.sharpness_label.pack(expand=1, anchor=W)

        self.sharpness_scrollbar = Scale(self.par_frame, bg='#fffff0', cursor='hand1', from_=0, to=50, orient=HORIZONTAL, showvalue=1, variable=self.sharpness, relief=FLAT, length=172, command=self.process_image)
        self.sharpness_scrollbar.pack(expand=1, anchor=W)

        # ------------------------------------------------ Место для фото ------------------------------------------------
        self.img_frame = LabelFrame(root, text='', bg='#fffff0', bd=1, height=465, width=740)
        self.img_frame.place(relx=0.25, rely=0.04)

        self.img_canvas = Canvas(self.img_frame, bg='#fffff0', bd=1, height=465, width=740)
        self.img_canvas.place(relx=0, rely=0)

        # --------------------------------------------------------------- PSF -----------------------------------------------------------
        self.psf_frame = LabelFrame(root, text='PSF', bg='#fffff0', bd=1, fg='#696969', font='Courier 16', height=172, width=172)
        self.psf_frame.place(relx=0.025, rely=0.55)

        self.psf_img = Canvas(self.psf_frame, bg='#fffff0', bd=1, height=172, width=172)
        self.psf_img.place(relx=0, rely=0)

        # ------------------------------------------------ О программе ------------------------------------------------
        self.caption = "Welcome!\nApp was created\nby Irina Skurko\nGroup 814301\nBSUIR 2019"
        self.about_label_frame = LabelFrame(root, text=self.caption, bg='#fffff0', bd=0.5, fg='#696969', font='Courier 12', height=120, width=172)
        self.about_label_frame.place(relx=0.025, rely=0.8)

        # ------------------------------------------------ Кнопки ------------------------------------------------ 
        self.load_img_button = Button(text='  Load  image  ', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.input_img)
        self.load_img_button.place(relx=0.3, rely=0.8)

        self.show_orig_button = Button(text=' Show original ', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.show_start_image)
        self.show_orig_button.place(relx=0.54, rely=0.8)

        self.show_result_button = Button(text=' Show result ', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.show_result_image)
        self.show_result_button.place(relx=0.78, rely=0.8)

        self.show_orig_button = Button(text='Save image', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.save_result)
        self.show_orig_button.place(relx=0.35, rely=0.9)

        self.pfc_button = Button(text='  Spectrum  ', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.spectrum)
        self.pfc_button.place(relx=0.55, rely=0.9)

        self.hist_button = Button(text='Histogram', bg='#E7D7D3', activebackground='#F5E5E2', font='Courier 16', command=self.hist)
        self.hist_button.place(relx=0.8, rely=0.9) 

    # Поместить иллюстрацию с PSF в Canvas
    def process_image(self, par):
        self.angle = self.angle_scrollbar.get()
        self.radius = self.radius_scrollbar.get()
        self.sharpness = self.sharpness_scrollbar.get()
        self.new_blur_type = self.blur_type.get()

        self.furier_img = cv2.dft(self.cv_img, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.noise = 10**(-0.1*self.sharpness)
        if self.new_blur_type == 1:
            self.psf = Deconvolution.defocus_kernel(self.radius)
        else:
            self.psf = Deconvolution.motion_kernel(self.angle, self.radius)
        psf= self.psf

        _psf = Image.fromarray(self.psf*255)
        result_psf = _psf.resize((160, 160), Image.ANTIALIAS)
        self.psf_pic = ImageTk.PhotoImage(result_psf)
        self.psf_img.create_image(80, 80, image=self.psf_pic)
        self.psf_img.update()
        
        psf /= psf.sum()
        psf_pad = np.zeros_like(self.cv_img) 
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf

        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + self.noise)[...,np.newaxis]

        RES = cv2.mulSpectrums(self.furier_img, iPSF, 0)
        self.res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        self.res = np.roll(self.res, -kh//2, 0) #Элементы, которые выходят за пределы последней позиции, повторно вводятся при первой
        self.res = np.roll(self.res, -kw//2, 1)

        self.spectrum_res = self.res

        res_image = Image.fromarray(self.res*255)
        self.image = ImageTk.PhotoImage(res_image)
        self.result_image = self.image

        self.img_canvas.create_image(350, 200, image=self.image)
        self.img_canvas.update()
    def input_img(self):
        self.path = askopenfilename(filetypes = (("JPG image", "*.jpg"),("PNG image", "*.png"),("ALL files", "*.*") ))
        image = Image.open(self.path)
        image = image.convert('L')
        h = image.height
        w = image.width
        if h > 500 or w > 800:
            if h < w:
                koef = int(h / 465);
            else:
                koef = int(w / 740)
            image = image.resize((int(h/koef), int(w/koef)), Image.ANTIALIAS)
        self.cv_img = np.float32(image)/255.0

        image = Image.fromarray(self.cv_img*255)
        self.image = ImageTk.PhotoImage(image)

        self.start_image = self.image
        self.img_canvas.delete("all")
        self.img_canvas.create_image(350, 200, image=self.image)
        self.cv_img = Deconvolution.blur_edge(self.cv_img)
        self.spectrum_image = self.cv_img

    def show_start_image(self):
        self.image = self.start_image
        self.img_canvas.create_image(350, 200, image=self.image)
        self.img_canvas.update()

    def show_result_image(self):
        self.image = self.result_image
        self.img_canvas.create_image(350, 200, image=self.image)
        self.img_canvas.update()

    def save_result(self):
         plt.imshow(self.spectrum_res, cmap = 'gray')
         plt.xticks([]), plt.yticks([])
         plt.show()

    def spectrum(self):
        input_img = self.spectrum_image
        dft_input_img = cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)
        plt.subplot(221),plt.imshow(input_img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        
        image_shift = np.fft.fftshift(dft_input_img)
        magnitude_spectrum = 20*np.log(cv2.magnitude(image_shift[:,:,0], image_shift[:,:,1]))
        plt.subplot(222), plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        plt.subplot(223),plt.imshow(self.res, cmap = 'gray')
        plt.title('Output Image'), plt.xticks([]), plt.yticks([])

        img = self.spectrum_res
        furier_img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        image_shift = np.fft.fftshift(furier_img)
        magnitude_spectrum = 20*np.log(cv2.magnitude(image_shift[:,:,0], image_shift[:,:,1]))
        plt.subplot(224), plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        plt.show()

    def hist(self):
        plt.hist(abs(self.spectrum_res))
        plt.title("Histogram")
        plt.show()

class Deconvolution(MainWindow):
    def blur_edge(cv_img):
        bord=32
        h, w  = cv_img.shape[:2]
        img_pad = cv2.copyMakeBorder(cv_img, bord, bord, bord, bord, cv2.BORDER_WRAP)
        img_blur = cv2.GaussianBlur(img_pad, (2*bord+1, 2*bord+1), -1)[bord:-bord,bord:-bord]
        y, x = np.indices((h, w))
        dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
        w = np.minimum(np.float32(dist)/bord, 1.0)
        return cv_img*w + img_blur*(1-w)

    def motion_kernel(angle, d):
        sz=50
        angle = np.deg2rad(angle)
        kern = np.ones((1, d), np.float32)
        c, s = np.cos(angle), np.sin(angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = sz // 2
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
        kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
        return kern

    def defocus_kernel(d):
        sz = 65
        kern = np.zeros((sz, sz), np.uint8)
        cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
        kern = np.float32(kern) / 255.0
        return kern

root = Tk()
root.configure(background='#fffff0')
root.geometry("1020x720+300+300")
root.title('Image restoration')
Style().configure("TFrame", background='#fffff0')  
window = MainWindow()

root.mainloop()