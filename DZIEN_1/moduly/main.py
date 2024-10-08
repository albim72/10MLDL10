# import dane
# import dane as dn
from dane import nrfilii as nf
from dane import book as bk

from moje_funkcje.collectfunction import czytaj_liste, czytaj_slownik

print("kolekcja z modułu: dane")
print(nf)
print(bk)

print("kolekcje wyświetlane funkcjami:")
print("_"*50)
czytaj_liste(nf)
print("_"*50)
czytaj_slownik(bk)
