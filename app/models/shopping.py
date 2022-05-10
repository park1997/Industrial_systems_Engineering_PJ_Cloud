import collections
from odmantic import Model

class ShoppingModel(Model):
    keyword : str
    title : str
    price : int
    image : str
    category : str
    brand : str
    link : str
    
    class Config:
        collection = "shop"
        