# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:46:29 2018

@author: leeyo
"""
class MyCircularQueue:

    #Constructor
    def __init__(self, k):
        self.queue = [None] * k
        self.head = 0
        self.tail = 0
        self.maxSize = k

    #Adding elements to the queue
    def enqueue(self,data):
        if self.size() == self.maxSize-1:
            return ("Queue Full!")
        self.queue[self.tail] = data
        self.tail = (self.tail + 1) % self.maxSize
        return True

    #Removing elements from the queue
    def dequeue(self):
        if self.size()==0:
            return ("Queue Empty!") 
        data = self.queue[self.head]
        self.head = (self.head + 1) % self.maxSize
        return data

    #Calculating the size of the queue
    def size(self):
        if self.tail>=self.head:
            return (self.tail-self.head)
        return (self.maxSize - (self.head-self.tail))