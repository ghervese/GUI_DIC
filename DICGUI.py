import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import filedialog
import os
from matplotlib.patches import Rectangle
import muDIC as dic
from PIL import Image, ImageDraw
from matplotlib import image as mpimg
from matplotlib.pyplot import connect
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.widgets  import RectangleSelector

#######################################################################################################################################
#                                            User friendly GUI for Digital Image Correlation with muDIC   
#                                    V0.1 by Guillaume Hervé-Secourgeon // guillaume.herve-secourgeon[at]edf.fr
#######################################################################################################################################
# The purpose of this Python class is to provide an environment easing the use of muDIC and the pre-processing of a set of images
class muDIC_GUI:
    """
    This class is a GUI dedicated to 3 purposes:
    - Pre-process a set of experimental images
    - Carry out Digital Image Correlation based on muDIC class.
    - Post-process the results
    - Carry out comparisons with FEM computations results

    This class uses tkinter that is coming naturally with Python installations.
    The core of the class relies on muDIC class.
    Caution: for Ubuntu environment the muDIC class has to be installed manually and not through pip
    For theoretical details regarding muDIC see: https://www.sciencedirect.com/science/article/pii/S2352711019301967 
    (muDIC: An open-source toolkit for digital image correlation, S.N. Olufsen et al., SoftwareX Volume 11, January–June 2020, 100391)
    muDIC is a non-local FE based Digital Image Correlation class. It is entirely developed in Python.
    The other prerequisites are PIL, Numpy and MatPlotLib.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("User Interface for pre-processing images and DIC based on muDIC class")
        self.fig_photo_first = None
        self.rect = None

    plt.rcParams['figure.figsize'] = [0.1, 0.1]

    def Close(self):
        """
        Close the GUI and the app
        """
        self.root.destroy()    
    def run(self):
        """
        Launch the GUI
        This method initiates the principal loop of the tkinter based GUI and the app on screen.

        """
        self.create_gui()
        self.root.mainloop()

    def select_images_folder(self):
        """
        This method is a dialog box to get the path accessing to the input images
        """

        self.source_selection_path = filedialog.askdirectory(title='Select the Directory hosting the images')
   
    def select_output_folder(self):
        """
        This method is a dialog box to get the path for output results
        """
        self.output_selection_path = filedialog.askdirectory(title='Select the Output Directory')

    def get_corner1(self,event):
        # print('you pressed for coord of point :', event.button, np.round(event.xdata), np.round(event.ydata))
        self.corner1=(np.array([np.round(event.xdata),np.round(event.ydata)]))
        # print(self.coord_window)
#         if len(self.coord_window)==2:
# #            self.center_ROI = np.round(np.abs((self.coord_window[1]-self.coord_window[0]))/2.)
#             self.longx_ROI = np.round(np.abs((self.coord_window[1][0]-self.coord_window[0][0])))
#             self.longy_ROI = np.round(np.abs((self.coord_window[1][1]-self.coord_window[0][1])))
#             # r = Rectangle(tuple(self.center_ROI), self.longx_ROI, self.longy_ROI,
#             #     edgecolor='red', facecolor='none')
#             self.preview_selection.subplots().add_patch(Rectangle(tuple(self.coord_window[0]),width=self.longx_ROI,height=self.longy_ROI, edgecolor="red", fill=False))
    def get_corner2(self,event):
        # print('you pressed for coord of point :', event.button, np.round(event.xdata), np.round(event.ydata))
        self.corner2=(np.array([np.round(event.xdata),np.round(event.ydata)]))


    # def line_select_callback(self,eclick, erelease):
    #     x1, y1 = eclick.xdata, eclick.ydata
    #     x2, y2 = erelease.xdata, erelease.ydata

    #     rect = Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) ,facecolor='None',edgecolor='Red')
    #     print(rect.get_corners())
    #     ax=self.preview_selection.gca()
    #     ax.add_patch(rect)




    def line_select_callback(self,eclick, erelease):
        plt.gcf()
        plt.close('all')

        print('entre dans la selection')
        # x1, y1 = eclick.xdata, eclick.ydata
        # x2, y2 = erelease.xdata, erelease.ydata
        self.coin_1 = np.round([eclick.xdata, eclick.ydata])
        self.coin_2 = np.round([erelease.xdata, erelease.ydata])

    #    rect = Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) ,facecolor='None',edgecolor='Red')
    #     print(rect.get_corners())
    #     ax=self.preview_selection.gca()
    #    self.ax.add_patch(rect)
        print('First corner coordinates in px:',self.coin_1)
        print('Second corner coordinates in px:',self.coin_2)

    def select_ROI_rectangle(self):
        plt.close('all')
        plt.gcf()

        rs = RectangleSelector(self.ax, self.line_select_callback,useblit=False, button=[1], 
                            minspanx=5, minspany=5, spancoords='pixels', 
                            interactive=True)

        plt.show()



    def select_menu_num_images(self,event):
        self.num_images = self.list_combo_num_images.get()

    def select_menu_format_image(self,event):
        self.format_image = self.list_combo_format_image.get()


    def first_image_view(self):
        """
        This method loads the first image in the selected directory with the proper prefix, numbering and format.
        It converts it into a hypermatrix with pixels in row and columns and 4 digits corresponding to the Bayer matrix value
        If the image is in true gray level, this is not a hypermatrix but a "simple" matrix with 1 digit for each pixel location.
        """
        # image = self.source_selection_path +'/' +self.prefix_entry.get() + str(self.num_images) + '1' + self.format_image
        # image = mpimg.imread(image)
        self.preview = mpimg.imread(self.source_selection_path +'/' +self.prefix_entry.get() + str(self.num_images) + '1' + self.format_image)

        plt.clf()
        self.preview_selection = plt.figure(figsize=(15,2))
#        self.ax = self.preview_selection.add_subplot()
        self.ax = self.preview_selection.subplots()
        self.ax.imshow(self.preview,cmap='binary')
        self.ax.grid(color='black',ls='solid')

        if self.fig_photo_first is not None:
            self.fig_photo_first.get_tk_widget().destroy()
        # self.fig_photo_first = FigureCanvasTkAgg(self.preview_selection, master=self.canvas_FOV_ROI)
        self.fig_photo_first = FigureCanvasTkAgg(self.preview_selection, master=self.canvas_FOV_ROI)
        self.fig_photo_first.draw()
        self.fig_photo_first.get_tk_widget().pack()

        # self.zone_select_ROI = tk.Canvas(self.fig_photo_first)
        # self.zone_select_ROI.bind("<Button-1>", self.on_button_pressed)
        # self.zone_select_ROI.mainloop()
        # self.fig_photo_first.bind("<Button-1>", self.on_button_pressed)
        # self.fig_photo_first.mainloop()


    def plot_ROI_on_fig(self):
        self.corner1 = np.array(self.corner1_entry.get())
        self.corner2 = np.array(self.corner2_entry.get())
        print(np.shape(self.corner1))
        print(np.shape(self.corner2))
        # print((self.corner2*2.))
        # r = Rectangle(tuple([self.tile_coordinates_center[i].ra.degree,self.tile_coordinates_center[i].dec.degree]), self.tile_fov*1.4, self.tile_fov,
        # edgecolor='red', facecolor='none',
        # transform=ax.get_transform('world')
        # )
        # r = Rectangle(tuple(AJOUTER LES COORDONEES DU CENTRE A CALCULER), LARGEUR, LONGUEUR,
        # edgecolor='red', facecolor='none')
        # self.fig_photo_first.add_patch(r)
  


    def create_gui(self):
        """
        Creates the GUI

        This method uses the tkinter class to generate the different graphical elements of the GUI: buttons, text areas and tabs.
        It calls the different methods of the muDIC_GUI class
        """
        # Pre-processing of images
        tab_control = ttk.Notebook(self.root)
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='Pre-processing')
        tab_control.pack(expand=1, fill='both')

        # Definition of the ROI, creation of the mesh and calculation
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='Digital Image Correlation')
        tab_control.pack(expand=1, fill='both')

        # Post-processing of the different quantities of interest
        tab3 = ttk.Frame(tab_control)
        tab_control.add(tab3, text='Post-processing')
        tab_control.pack(expand=1, fill='both')

        ##########################################################
        # Code relatif au premier onglet
        ##########################################################

        # Frame pour user data
        preprocessing_frame = ttk.LabelFrame(tab1, text='Pre-processing')
        preprocessing_frame.pack(expand=1, fill='both', padx=2, pady=2)

        chose_path_button = ttk.Button(preprocessing_frame, text="Select the location folder of the images", command=self.select_images_folder)
        chose_path_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        quit_button = ttk.Button(preprocessing_frame,text = "Quit", command = self.Close,width=5)
        quit_button.grid(row=0, column=6,columnspan=6, padx=5, pady=5)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Prefix of the images (replace the example):')
        object_label.grid(row=1, column=1, padx=5, pady=5)
        self.prefix_entry = ttk.Entry(preprocessing_frame)
        self.prefix_entry.insert(0, 'Gauche_Droite-')
        self.prefix_entry.grid(row=1, column=2, padx=5, pady=5)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Suffix of the images (replace the example):')
        object_label.grid(row=2, column=1, padx=5, pady=5)
        self.suffix_entry = ttk.Entry(preprocessing_frame)
        self.suffix_entry.insert(0, '_gauche')
        self.suffix_entry.grid(row=2, column=2, padx=5, pady=5)


        # Menu to select the numbering of the stack of images
        num_images = ttk.Label(preprocessing_frame, text='Numbering of the images (ex: if 00 selected then 00x):')
        num_images.grid(row=1, column=3, padx=5, pady=5)

        # list of the supported numbering types
        self.list_num_images = ['0','00','000','0000','00000','000000']
        # creation comboBox
        self.list_combo_num_images=ttk.Combobox(preprocessing_frame, values=self.list_num_images,width=6)
        default_num_image = 3
        self.list_combo_num_images.current(default_num_image)
        #ComboBox location
        self.list_combo_num_images.grid(row=1, column=4, padx=5, pady=5)
        self.list_combo_num_images.bind("<<ComboboxSelected>>",self.select_menu_num_images)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.num_images = self.list_num_images[default_num_image]

        # Menu to select the numbering of the stack of images
        format_image = ttk.Label(preprocessing_frame, text='Format of the images:')
        format_image.grid(row=2, column=3, padx=5, pady=5)

        # list of the supported numbering types
        self.list_format_image = ['.tif','.tiff','.png']
        # creation comboBox
        self.list_combo_format_image=ttk.Combobox(preprocessing_frame, values=self.list_format_image,width=5)
        default_format_image = 1
        self.list_combo_format_image.current(default_format_image)
        #Position de la ComboBox
        self.list_combo_format_image.grid(row=2, column=4, padx=5, pady=5)
        self.list_combo_format_image.bind("<<ComboboxSelected>>",self.select_menu_format_image)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.format_image = self.list_format_image[default_format_image]

        import_first_image_button = ttk.Button(preprocessing_frame,text = "View first image", command = self.first_image_view)
        import_first_image_button.grid(row=2, column=5,columnspan=1, padx=5, pady=5)



        # Frame showing the first images of the stack
        self.preview_first_image_frame = ttk.LabelFrame(tab1, text='Field of view and ROI selection')
        self.preview_first_image_frame.pack(expand=0, fill='both', padx=2, pady=2)

        # Canvas pour afficher l'aperçu
        self.canvas_FOV_ROI = tk.Canvas(self.preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_FOV_ROI.pack()

        ROI_frame = ttk.LabelFrame(tab1, text='Definition of the Region of Interest')
        ROI_frame.pack(expand=1, fill='both', padx=2, pady=2)

        select_ROI_button = ttk.Button(ROI_frame,text = "On-screen selection of the ROI", command = self.select_ROI_rectangle)
        select_ROI_button.grid(row=0, column=0,columnspan=2, padx=5, pady=5)

        show_ROI_button = ttk.Button(ROI_frame,text = "Show selected ROI", command = self.plot_ROI_on_fig)
        show_ROI_button.grid(row=2, column=0,columnspan=2, padx=5, pady=5)

        resize_ROI_button = ttk.Button(ROI_frame,text = "Resize images to fit ROI", command = self.Close)
        resize_ROI_button.grid(row=2, column=3,columnspan=2, padx=5, pady=5)


# Instanciation de l'application et exécution
app = muDIC_GUI()
app.run()
