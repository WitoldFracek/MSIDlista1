import numpy as np

#basic
a = np.array([1, 2, 3])
print(a)
a = np.zeros(3)
print(a)
a = np.empty(5)
print(a)
a = np.arange(4)
print(a)
a = np.arange(2, 100, 20) #start, end, step
print(a)
a = np.linspace(0, 9, 1)  #zacznij od 0, skoncz na 9, krok taki zeby byl 1 element
print(a)
a = np.linspace(0, 9, 10) #zacznij od 0, skoncz na 9, krok taki zeby bylo 10 elementow
print(a)
a = np.linspace(0, 9, 5) #zacznij od 0, skoncz na 9, krok taki zeby bylo 5 elementow
print(a)
a = np.ones(3, dtype=np.int64)
print(a)
a = np.concatenate((a, a))
print(a)


#dimentions
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.ndim(a)) #Ile wymiarow
print(np.size(a)) #Ile wartosci
print(np.shape(a)) #Jakie rozmiary


#reshape
a = np.arange(6)
b = a.reshape(3, 2)
c = a.reshape(6, 1)
print(a)
print(b)
print(c)

#adding more axis
a = np.arange(6)
a2 = a[np.newaxis, :]
print(a2)
a2 = a[:, np.newaxis]
print(a2)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a < 7])
less_than_7 = (a < 7)
print(a[less_than_7])

#dostep
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)
print()
print(a[0, 0])
print()
print(a[:, 0])
print()
print(a[1, :])
print()
print(a[1:, :])
print()





