#%%
from PIL import Image
import numpy as np
import cv2
import random
import math
import json

#Funktionen
#Bild skalieren
def scale_image(image, scale_factor_range):
            #Array als Bild umwandeln
            image = Image.fromarray(image)
            original_size = np.array(image.size)
            #Skalierungsparameter laden
            new_scale_factor_x = random.uniform(*scale_factor_range)
            new_scale_factor_y= random.uniform(*scale_factor_range)
            #Neue Dimensionen berechnen
            new_width = int(original_size[0] * new_scale_factor_x)
            new_height = int(original_size[1] * new_scale_factor_y)
            new_size = (new_width, new_height)
            #Bild vergrößern
            image = image.resize(new_size, resample=Image.BICUBIC)
            image = np.array(image)
            return image, new_scale_factor_x, new_scale_factor_y

#Bild rotieren
def rotate_image(image, angle):
    #Mittelpunkt des Bildes berechnen
    rows, cols = image.shape[:2]
    rotation_center = (cols/2, rows/2)
    #Rotation um den Mittelpunkt mittels Rotationsmatrix
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    #neue Größe des Bildes berechnen, damit nichts abgeschnitten wird
    nrows = int((rows * cos) + (cols * sin))
    ncols = int((rows * sin) + (cols * cos))
    rotation_matrix[0, 2] += (ncols / 2) - (cols / 2)
    rotation_matrix[1, 2] += (nrows / 2) - (rows / 2)
    rotated = cv2.warpAffine(image, rotation_matrix, (ncols, nrows))

    #schwarzen Rand nach Rotation entfernen
    #Kontour des rotierten Kraters finden
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Maske der Kontour erstellen
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=1)
    mask = cv2.bitwise_not(mask)

    #schwarzen Rand Alpha = 0 setzen (durchsichtig)
    rotated_with_border = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
    rotated_with_border[mask == 0] = [0, 0, 0, 0]
    return rotated_with_border, rotation_center

#Krater in den Hintergrund einfügen
def insert_foreground_with_mask(background, foreground, mask, x, y, background_mask):
    foreground_height, foreground_width = foreground.shape[:2]
    while True:
        #check, ob Platz im Hintergrund frei ist
        occupied = False
        for i in range(y, y + foreground_height):
            for j in range(x, x + foreground_width):
                if background_mask[i, j]:
                    occupied = True
                    break
            if occupied:
                break
        if not occupied:
            break
        #wenn nicht, dann neue Koordinaten zum Einfügen des Kraters
        x = np.random.randint(0, background.shape[1] - foreground.shape[1])
        y = np.random.randint(0, background.shape[0] - foreground.shape[0])
        #Krater einfügen und Maske updaten
    for i in range(foreground_height):
        for j in range(foreground_width):
            if mask[i, j]:
                background[y + i, x + j] = foreground[i, j]
                #occupied = True für belegte Pixelkoordinaten
                background_mask[y + i, x + j] = True
    return background, background_mask, x, y

#Artefakte in den Hintergrund einfügen
def insert_artefact_with_mask(background, artefact, mask, x, y, background_mask):
    artefact_height, artefact_width = artefact.shape[:2]
    #check, ob Platz im Hintergrund frei ist
    while True:
        occupied = False
        for i in range(y, y + artefact_height):
            for j in range(x, x + artefact_width):
                if background_mask[i, j]:
                    occupied = True
                    break
            if occupied:
                break
        if not occupied:
            break
        #wenn nicht, dann neue Koordinaten zum Einfügen des Artefakts
        x = np.random.randint(0, background.shape[1] - artefact.shape[1])
        y = np.random.randint(0, background.shape[0] - artefact.shape[0])
    #Artefakt einfügen und Maske updaten
    for i in range(artefact_height):
        for j in range(artefact_width):
            if mask[i, j]:
                background[y + i, x + j] = artefact[i, j]
                #occupied = True für belegte Pixelkoordinaten
                background_mask[y + i, x + j] = True
    return background, background_mask, x, y

#Maske erzeugen Krater
def create_mask(image):
    #Maske mit Dimension des Kraterbildes erstellen
    alpha_channel = image[..., 3]
    mask = np.zeros_like(alpha_channel, dtype=bool)
    mask[alpha_channel > 0] = True
    return mask

#Maske erzeugen Artefakt
def create_mask_artefact(image):
    #Maske mit Dimension des Kraterbildes erstellen
    alpha_channel = image[..., 3]
    mask_artefact = np.zeros_like(alpha_channel, dtype=bool)
    mask_artefact[alpha_channel > 0] = True
    return mask_artefact

#Color Jitter hinzufügen
def color_jitter(img, brightness_factor, contrast_factor, saturation_factor):
    #in RGB
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    #Werte für die Manipulation definieren
    brightness = random.uniform(brightness_factor[0], brightness_factor[1])
    contrast = random.uniform(contrast_factor[0], contrast_factor[1])
    saturation = random.uniform(saturation_factor[0], saturation_factor[1])
    
    #Manipulation anwenden und zurück konvertieren
    img = img * contrast + brightness
    img = np.clip(img, 0, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 1] *= saturation
    img = np.clip(img, 0, 1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = (img * 255.0).astype(np.uint8) 
    return img

#Bild in Graustufen
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#Boundry Box an Transformation anpassen
def adjust_bbox(original_bbox, new_scale_factor_x, new_scale_factor_y, angle, x, y, image_shape, rotation_center):
    #Boundry Box skalieren
    bbox_points = original_bbox.copy()
    bbox_points[:, 0] *= new_scale_factor_x
    bbox_points[:, 1] *= new_scale_factor_y
    
    #Rotationvariablen initiieren
    axis = rotation_center
    rect = bbox_points
    
    #Distanz zum Rotationszentrum berechnen
    tx = -axis[0] 
    ty = -axis[1]
    
    #Boundry Box auf das Rotationszentrum beziehen
    rect = [(p[0] + tx, p[1] + ty) for p in rect]
    
    #Rechteck mittels Rotationsmatrix um das Rotationszentrum rotieren
    theta = -math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    rect = [(c * p[0] - s * p[1], s * p[0] + c * p[1]) for p in rect]

    #Mittelpunkt des Bildes berechnen
    tx = -image_shape[1]/2
    ty = -image_shape[0]/2

    #Rechteck wieder auf die obere linke Ecke des Bildes beziehen
    rect = [(p[0] - tx, p[1] - ty) for p in rect]
    rect = np.round(np.array(rect)).astype(int)
    bbox_points = rect

    #neue Boundry Box definieren
    x_min = np.min(bbox_points[:, 0]) +x
    y_min = np.min(bbox_points[:, 1]) +y
    x_max = np.max(bbox_points[:, 0]) +x
    y_max = np.max(bbox_points[:, 1]) +y
    bbox = {"left": (int(x_min)), 
            "top": (int(y_min)), 
            "width": (int(x_max)- int(x_min)), 
            "height": (int(y_max) - int(y_min))}
    return bbox

#Polygon / Maske anpassen
def adjust_polygon(polygon, new_scale_factor_x, new_scale_factor_y, angle, x, y, image_shape, rotation_center):
    #Boundry Box skalieren
    polygon_points = np.array([[point['x'], point['y']] for point in polygon])
    
    #Polygon skalieren
    polygon_points[:, 0] *= new_scale_factor_x
    polygon_points[:, 1] *= new_scale_factor_y
    
    #Rotationvariablen initiieren
    axis = rotation_center
    rect = polygon_points
    
    #Distanz zum Rotationszentrum berechnen
    tx = -axis[0] 
    ty = -axis[1]
    
    #Polygon auf das Rotationszentrum beziehen
    rect = [(p[0] + tx, p[1] + ty) for p in rect]
    
    #Polygon mittels Rotationsmatrix um das Rotationszentrum rotierenn
    theta = -math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    rect = [(c * p[0] - s * p[1], s * p[0] + c * p[1]) for p in rect]

    #Mittelpunkt des Bildes berechnen
    tx = -image_shape[1]/2
    ty = -image_shape[0]/2

    #Polygon wieder auf die obere linke Ecke des Bildes beziehen
    rect = [(p[0] - tx, p[1] - ty) for p in rect]
    polygon_points_rotated = np.round(np.array(rect), decimals=2).astype(float)
    
    #neues Polygon definieren
    adjusted_polygon = [{"x": float(point[0] + x), "y": float(point[1] + y)} for point in polygon_points_rotated]

    #Format konvertieren
    segmentation = []
    for point in adjusted_polygon:
        segmentation.append(point["x"])
        segmentation.append(point["y"])
    segmentation = [segmentation]
    return adjusted_polygon, segmentation

#Polygon Flächeninhalt
def polygon_area(polygon):
    #Shoelace Formel zur Berechnung
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i]['x'] * polygon[j]['y'] - polygon[j]['x'] * polygon[i]['y']
    return abs(area) / 2.0

#Salt und Pepper hinzufügen
def salt_and_pepper(img):
  
    #Größe des Bildes
    row , col = img.shape
      
    #Anzahl der Salzkörner
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        
        #Zufällige Koordinaten wählen
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
          
        #Pixel weiß setzen
        img[y_coord][x_coord] = 255
          
    #Anzahl der Pfefferkörner
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        
        #Zufällige Koordinaten wählen
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
          
        #Pixel schwarz setzen
        img[y_coord][x_coord] = 0
          
    return img


#Distanzformel
def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

#Definition Gaussian Blur
def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = math.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

#Fourier-Transformation
def apply_gaussian_blur(img, d0):
    img_fft = np.fft.fft2(img)
    img_fftshift = np.fft.fftshift(img_fft)
    lp_center = img_fftshift * gaussianLP(d0, img.shape)
    lp = np.fft.ifftshift(lp_center)
    inverse_lp = np.fft.ifft2(lp)
    return np.abs(inverse_lp).astype(np.uint8)

def gray_to_rgb(gray_image):
    # Shape des Graustufenbildes abrufen
    height, width = gray_image.shape

    # Neues Bild mit 3 Kanälen erstellen
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Den gleichen Wert für alle drei Kanäle setzen
    rgb_image[:,:,0] = gray_image
    rgb_image[:,:,1] = gray_image
    rgb_image[:,:,2] = gray_image

    return rgb_image

#Variabeln
AnzahlBilder = 1000
xName = 0
i = 1
annotations = []
images = []

#Maßstab defineren
dimSkalierung = (144, 30)
imgSkalierung = cv2.imread('C:/Users/maxmi/Desktop/Bachelorarbeit/Datenbanken/Datenbank Skala/Skalierung.png', cv2.IMREAD_UNCHANGED)
imgSkalierung = cv2.resize(imgSkalierung, dimSkalierung)

#Schleife für Datensynthese
for i in range(AnzahlBilder):
    #Initalisierung Datenbank
    polygons = []
    bboxes = []
    eIndex = []
    background_widths = []
    background_heights = []
    DropletNumberList = []
    segmentations = []
    areas = []
    #Zugriff Datenbank Hintergrund
    #Zufälliges Hintergrundbild aus der Datenbank auswählen
    randomBackgroundNumber = random.randint(1, 34)
    randomPathfileBackground = 'C:/Users/maxmi/Desktop/Bachelorarbeit/Datenbanken/Datenbank Hintergruende' + '/' + str(randomBackgroundNumber) +'.png'
    imgHintergrund = cv2.imread(randomPathfileBackground, cv2.IMREAD_UNCHANGED)
    #Bild an die gewünschte Größe anpassen
    width = 1920
    height = 1440
    dim = (width, height)
    imgBackgroundResize = cv2.resize(imgHintergrund, dim, interpolation = cv2.INTER_CUBIC)
    #Hintergrund randomisieren
    np.random.seed(i)
    np.random.shuffle(imgBackgroundResize)
    #Anzahl Krater erzeugen
    e = 0
    numberCraters = random.randint (5, 50)
    background = imgBackgroundResize.copy()
    #Hintergrund Maske erstellen
    background_mask = np.zeros_like(background[..., 0], dtype=int)
    for e in range(numberCraters):
        #Kraterbild aus der Datenbank laden
        randomDropletNumber = random.randint(1, 1000)
        randomPathfileCrater = 'C:/Users/maxmi/Desktop/Bachelorarbeit/Datenbanken/Datenbank Tropfen' + '/' + str(randomDropletNumber) +'.png'
        foreground = cv2.imread(randomPathfileCrater, cv2.IMREAD_UNCHANGED)

        #Parameter für Skalierung des Tropfen
        scale_factor_range = (0.80, 1.10)
        rgba_image, new_scale_factor_x, new_scale_factor_y  = scale_image(foreground, scale_factor_range)
        foreground = rgba_image

        #Zufällige Rotation des Tropfen
        angle=random.randint(1, 360)
        foreground,rotation_center = rotate_image(foreground, angle=angle)

        #Koordinaten zum Einfügen des Kraters in den Hintergrund
        x = np.random.randint(0, background.shape[1]-foreground.shape[1])
        y = np.random.randint(0, background.shape[0]-foreground.shape[0])

        #Krater in den Hintergrund einfügen
        mask = create_mask(foreground)
        background, background_mask, x, y = insert_foreground_with_mask(background, foreground, mask, x, y, background_mask)
        #Boundry Box und Maske des Kraters laden
        with open("C:/Users/maxmi/Desktop/Bachelorarbeit/Label/labelsabsolut.json", "r") as file:
                data = json.load(file)
                index = next((xxx for xxx, d in enumerate(data) if d["External ID"] == str(randomDropletNumber)+".png"), None)
                objects = data[index]["Label"]["objects"]
                #Boundry Box an Transformation des Kraters anpassen
                for obj in objects:
                    bbox = obj.get('bbox')
                    if bbox:
                        top = bbox.get('top')
                        left = bbox.get('left')
                        height = bbox.get('height')
                        width = bbox.get('width')
                        original_bbox = np.array([[float(left), float(top)], [float(left) + float(width), float(top)], [float(left), float(top) + float(height)], [float(left) + float(width), float(top) + float(height)]])
                        original_bbox = original_bbox.astype(np.float32)
                        foreground_shape = foreground.shape
                        bbox = adjust_bbox(original_bbox, new_scale_factor_x, new_scale_factor_y, angle, x, y, foreground_shape, rotation_center)
                        bboxes.append(bbox)

                #Polygon an Transformation des Kraters anpassen
                for obj in objects:
                    polygon = obj.get('polygon')
                    if polygon:
                        polygon, segmentation = adjust_polygon(polygon, new_scale_factor_x, new_scale_factor_y, angle, x, y, foreground_shape, rotation_center)
                        polygons.append(polygon)
                        area = polygon_area(polygon)
                        DropletNumberList.append(randomDropletNumber)
                        segmentations.append(segmentation)
                        areas.append(area)

    #Label in das COCO Format überführen
    m = 0
    for z in range(len(bboxes)):
        bbox = bboxes[z]
        bbox = [bbox['left'], bbox['top'], bbox['width'], bbox['height']]
        segmentation = segmentations[z]
        area = areas[z]
        annotation = {
            "segmentation": segmentation,
            "id": random.randint(0,999999999999),
            "iscrowd": 0,
            "image_id": i,
            "category_id": 1,
            "bbox": bbox,
            "area": float(area)
        }
        annotations.append(annotation)
        m = m+1
    image = {
        "id": i,
        "file_name": str(i) + ".png",
        "width": background.shape[1],
        "height": background.shape[0],
        "license": 0
        }
    categorie = {
        "supercategory": "KraterSuper",
        "id": 1,
        "name": "krater"}
    info = {"description": "BA 2023 Dataset","url": "no_url","version": "1.0","year": 2023,"contributor": "Mielke","date_created": "2023"}
    licenses = {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}
    images.append(image)
    annotations.append(annotation)
    save_data = {"info":info,"licenses":[licenses],"images":images,"annotations":annotations,"categories":[categorie]}
    
    #Loop Counter für die Anzahl der Krater erhöhen
    e = e + 1
    
    #Artefakte einfügen
    ArtefactsEnabled = random.choice([False, True])
    NumberArtefacts = random.randint(1, 5)
    if ArtefactsEnabled == True:
        for f in range(NumberArtefacts):
            #Zufälliges Artefakt aus der Datenbank laden
            randomArtefactNumber = random.randint(1, 103)
            randomPathfileArtefact = 'C:/Users/maxmi/Desktop/Bachelorarbeit/Datenbanken/Datenbank Artefakte' + '/' + str(randomArtefactNumber) +'.png'
            imgArtefact = cv2.imread(randomPathfileArtefact, cv2.IMREAD_UNCHANGED)
            artefact = imgArtefact.copy()
            #Skalierung des Artefakts
            rgba_image, new_scale_factor_x, new_scale_factor_y = scale_image(artefact, scale_factor_range)
            artefact = rgba_image
            #Zufällige Rotation des Artefakts
            angle=random.randint(0, 360)
            artefact, rotation_center = rotate_image(artefact, angle=angle)
            #Koordinaten zum Einfügen der Artefakts in den Hintergrund
            x = np.random.randint(0, background.shape[1]-artefact.shape[1])
            y = np.random.randint(0, background.shape[0]-artefact.shape[0])
            #Artefakt einfügen
            mask_artefact = create_mask_artefact(artefact)
            background, background_mask, x, y = insert_artefact_with_mask(background, artefact, mask_artefact, x, y, background_mask)
            #Loop Counter für die Anzahl der Artefakte erhöhen
            f = f+1
    else:
        pass

    #JSON Datei speichern
    with open(f"C:/Users/maxmi/Desktop/Bachelorarbeit/Label/annotations.json", "w") as f:
        json.dump(save_data, f)

    #Boundry Boxes und Maske zeichnen
    for bbox in bboxes:
        #Punkte der Boundry Box extrahieren
        left = bbox['left']
        top = bbox['top']
        width = bbox['width']
        height = bbox['height']
        #Rechteck zeichnen
        #cv2.rectangle(background, (left, top), (left + width, top + height), (0, 0, 0), 2)
    for polygon in polygons:
        #Punkte des Polygons extrahieren
        polygon = np.array([[point['x'], point['y']] for point in polygon])
        #Polygon zeichnen
        #cv2.fillPoly(background, [np.int32(polygon)], (0, 0, 128, 128))

    #Color Jitter
    background = color_jitter(background, [-0.1, 0.1], [0.9, 1.1], [0.9, 1.1])

    #Bild in Graustufen überführen
    background = rgb2gray(background)

    #Stärke des Blurrings
    d0 = random.randint(130, 150)

    #Anwendung des Blurring / Fourier-Transformation
    background = apply_gaussian_blur(background, d0)

    #Maßstab einfügen
    imgSkalierungGray = rgb2gray(imgSkalierung)
    background[1400:1400+imgSkalierungGray.shape[0], 1766:1910:] = imgSkalierungGray

    #Salz und Pfeffer
    background = salt_and_pepper(background)
    
    #Filename und Speicherort festlegen
    filename = xName
    filename = str(filename)
    pathname = "C:/Users/maxmi/Desktop/Bachelorarbeit/Synthetische Daten"
    pathfile = pathname + '\\' + filename + ".png"
    xName = xName+1
    syntheticImg = Image.fromarray((np.abs(background)).astype(np.uint8))
    syntheticImg.save(pathfile)

# %%
