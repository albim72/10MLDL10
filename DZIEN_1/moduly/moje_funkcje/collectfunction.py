def czytaj_liste(lista):
    for i,j in enumerate(lista):
        print(f"numer indeksu elementu: {i}, wartość elementu: {j}")
        
def czytaj_slownik(slownik):
    for x,y in slownik.item():
        print(f"element słownika: klucz -> {x}: wartość: {y}")
