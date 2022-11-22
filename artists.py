#from honeybees.artists import Artists as HoneybeesArtists
from polygene.artists import BaseArtist

class Artists(BaseArtist): #HoneybeesArtists
    def __init__(self, model):
        BaseArtist.__init__(self, model)

    def draw_farmers(self, model, agents, idx, variable, color, minimum, maximum):
        elevation = getattr(agents, variable)[idx]
        alpha = self.get_alpha(elevation, 700,4000) #33000, 34000
        return {"type": "shape", "shape": "circle", "r": 1, "filled": True, "color": self.add_alpha_to_hex_color(color, alpha)}

    def draw_rivers(self):
        return {"type": "shape", "shape": "line", "color": "Blue"}