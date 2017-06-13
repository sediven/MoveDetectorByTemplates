# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = min(comm.Get_size(),7) #Nie ma więcej możliwych do użycia funkcji templatowych


def calculate_move(a,b):
    """Wyliczamy różnicę między punktami"""
    x_s = a[0] - b[0]
    y_s = a[1] - b[1]
    return ( x_s, y_s )

def get_mean(lista):
    """ Redukujemy wymiar listy do wspolrzednych """
    lb1 = []
    lt1 = []
    lb2 = []
    lt2 = []
    for i in lista:
        lb1.append(i[0][0])
        lt1.append(i[1][0])
        lb2.append(i[0][1])
        lt2.append(i[1][1])
    b = (int(np.mean(lb1)), int(np.mean(lb2)))
    t = (int(np.mean(lt1)),int( np.mean(lt2)))
    return b, t

template = cv2.imread('z90902.jpg', 0)
scale = 0.5 # dla mniejszej skali mniejsza skuteczność lepsze czasu przetwarzania
template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
ile= 100 #ile klatek chcemy przetworzyć

# print rank
if rank == 0:
    cap = cv2.VideoCapture(0)
    tmp_corner = (0L, 0L)
    plt.ion()

    start = 0
    for i in xrange(ile):

        ret, frame = cap.read() #pobieramy obraz z kamery, często peirwsza klatka jest pusta
        if frame is None:
            ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #mierzymy czas przetwarzania klatki bo na to mamy wpływ, wczytanie z kamery zależy od sprzętu i innych czynników
        start = time.time()
        if not start:
            start = time.time()
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        img = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        # comm.send([methods[1]], dest=1, tag=13)
        # comm.send(img, dest=1, tag=144)
#         print '2'
        chunk = len(methods) / (size - 1)
        # map
        for i in xrange(size - 1):
            begin = chunk * i
            end = begin + chunk
            comm.send(methods[begin:end], dest=i + 1, tag=13)
            comm.send(img, dest=i + 1, tag=144)
        #gather and reduce
        reduced_list = []
        for i in xrange(size - 1):
            temp = comm.recv(source=i + 1, tag=15)
            reduced_list.append(temp)
        bottom_right, top_left = get_mean(reduced_list)
#
        print calculate_move(tmp_corner, bottom_right)
        tmp_corner = bottom_right
        print bottom_right, top_left
        end = time.time()
        seconds = end - start
        print "Time taken : {0} seconds".format(seconds)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
#         # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    cap.release()
    cv2.destroyAllWindows()
#     plt.close()
#
elif rank>0 and rank <7:
    for i in xrange(ile):
        content = comm.recv(source=0, tag=13)
        img = comm.recv(source=0, tag=144)
        w, h = template.shape[::-1]
        lista =[]
        for m in content:
            method = eval(m)

            # Stosujemy matchowanie tamplate'a
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Jeśli TM_SQDIFF albo TM_SQDIFF_NORMED, musimy wziąć min
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            lista.append((bottom_right, top_left))
        if len( content )>1:
            bottom_right,top_left = get_mean(lista)
        comm.send((bottom_right,top_left), dest=0, tag=15)
