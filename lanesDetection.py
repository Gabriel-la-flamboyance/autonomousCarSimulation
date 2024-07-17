#####display image
import cv2
import numpy as np # for edge detection
import matplotlib.pyplot as plt # to help us to clarify how we are going to isolate a certain region

def make_coordinates(image, line_parameters): 
    slope, intercept = line_parameters 
    #print(image.shape) # will show 3 values "height, width and nb of channels"
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) # to make the coordinates start from the very bottom of the image and go upwards 3/5 of the way UP 
    x1 = int((y1 - intercept)/slope) 
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2]) 
        

def average_slope_intercept(image, lines): 
    left_fit = []
    right_fit = []
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
        parameters = np.polyfit((x1, x2),(y1, y2), 1) 
        #print(parameters) # for X line that we iterate through, it will print the slope and the Y intercept
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: 
            left_fit.append((slope, intercept))
        else: 
            right_fit.append((slope, intercept)) 
    #print(left_fit) # all slopes on the left side
    #print(right_fit) # all slopes on the right side
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0) 
    #print(left_fit_average, 'left') # slope of the left line
    #print(right_fit_average, 'right') # slope of the right line 
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line]) 

#we make a function for helping us to find region of interest
def canny(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #CHANGES INTO GRAY
    blur = cv2.GaussianBlur(gray, (5,5), 0) #we use "GaussianBlur" to reduce noise in our image (smoothens our image)
    canny = cv2.Canny(blur, 50,150) #to show a gradient image
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines: 
            #print(line) 
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            # draw the line, cordinates of the line, color, thickness of the line
    return line_image 

def region_of_interest(image):
    height = image.shape[0] 
    polygons = np.array([[(200, height), (1100, height), (550, 250)]]) #on créé un array of 1 polygon #quand on créé le graphe, une portion de la toute est entre 200 et 11OO, et là on trace un rectangle comme zone,
    mask = np.zeros_like(image)  
    cv2.fillPoly(mask, polygons, 255) #created a white triangle
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image 

################################ for images ################################

# image = cv2.imread('1.jpg') #will read our image 
# lane_image = np.copy(image) # we make a copy to avoid the changes on the original image
# canny_image = canny(lane_image) # will show the black image 
# cropped_image = region_of_interest(canny_image) # will onlyshow the lanes

# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# #                             2 pixels, degree precision, minimal votes to accept a candidate line, place holder array, length of line and pixels, distance btn line segments
# averaged_lines = average_slope_intercept(lane_image, lines)  
# line_image = display_lines(lane_image, averaged_lines) 
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # multiply elements with 0.8 to decrease picel intensity and makes images a little darker and for lines we x1 to make lines more visible

# ###
# ##we will change our image into greyscales bse as a coloured image have 3 channels "RGB" it takes time for processing while a greyimage has onlu one channel thus faster to process
# gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) #CHANGES INTO GRAY
# blur = cv2.GaussianBlur(gray, (5,5), 0) #we use "GaussianBlur" to reduce noise in our image (smoothens our image)
# #                         5x5 kernel (tableau) et la déviation '0'

# canny = cv2.Canny(blur, 50,150) #to show a gradient image
# #           montré les choses avec un gradient entre 50 et 150, ceux avec moins des gradients seront pas visible 
# cv2.imshow('result', image) #will show our image
# cv2.imshow('result', gray) # to see the gray image
# cv2.imshow('result', blur) # to see the blurled image
# cv2.imshow('result', canny) # to see a gradiented image (black image)
# ###
# cv2.imshow("result", combo_image)  
# # #plt.imshow(canny) # montré un graphe 
# cv2.waitKey(0) # will show the image for a certain time until we press a key (in our case infinitely bse od '0')
# plt.show() # will show the image on a grid of x&y axis   
# ################################ end ################################

#for videos
cap = cv2.VideoCapture("test3.mp4")  
while(cap.isOpened()): 
    _, frame = cap.read()
    canny_image = canny(frame) # will show the black image 
    cropped_image = region_of_interest(canny_image) # will onlyshow the lanes
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)  
    line_image = display_lines(frame, averaged_lines) 
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # multiply elements with 0.8 to decrease picel intensity and makes images a little darker and for lines we x1 to make lines more visible
    cv2.imshow("result", combo_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break
    
cap.release() 
cv2.destroyAllWindows() 