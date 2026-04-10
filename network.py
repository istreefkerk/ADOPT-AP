import numpy as np
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import coords_to_pixels
import pandas as pd
import os
import pyproj
import stats


class Network(AgentBaseClass):
    '''
    Class for (social) network and neighbourhood of the agents. Called apon in the Agents class.
    '''

#------------------------------------------------
#           LOCATION AND NEIGNBOURHOOD
#------------------------------------------------

    def location_index(self, pos):
        """
        Returns the x, y pixel locations for given coordinates.

        Args:
            pos (np.ndarray): Array of coordinates.

        Returns:
            tuple: Arrays of x and y pixel indices.
        """
        gt = self.gt
        return coords_to_pixels(pos, gt)

    def coords_neighbourhood(self, pos, radius):
        """
        Return a list of cells in the neighborhood of each agent.

        Args:
            pos (np.ndarray): Agent coordinates.
            radius (int or np.ndarray): Neighborhood radius (in cells).

        Returns:
            list: List of coordinate tuples for each agent's neighborhood.
        """
        coordinates_neighbourhood = []
        self.torus = True 
        gt = self.gt
        position = coords_to_pixels(pos, gt)

        for i in range(self.n): 
            co = []
            coordinates_neighbourhood.append(co)
            x, y = position[0][i], position[1][i]
            radius = int(self.radius[i])
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    coord = ( y + dy, x + dx)

                    if self.out_of_bounds(coord):
                    # Skip if not a torus and new coords out of bounds.
                        if not self.torus:
                            continue
                        coord = self.torus_adj(coord)
                    co.append(coord)
        return coordinates_neighbourhood

    def values_neighbourhood(self, coords, variable):
        """
        Get values of a variable at the coordinates in each agent's neighborhood.

        Args:
            coords (list): List of coordinate tuples.
            variable (np.ndarray): 2D array of variable values.

        Returns:
            list: List of lists of variable values for each agent.
        """
        array = []
        for i in range(self.n):
            values = []
            array.append(values)
            for j in range(len(coords[i])): #coords is size of agents, j gives tuple coordinates ()
                values.append(variable[coords[i][j]]) 
        return array

    def max_neighbourhood(self, coords, values):
        """
        Returns grid position of max value in neighborhood of each agent.

        Args:
            coords (list): Neighborhood coordinates.
            values (list): Neighborhood values.

        Returns:
            list: List of coordinate tuples with max value for each agent.
        """
        indices = []
        for i in range(self.n):
            indices.append(coords[i][values[i].index(max(values[i]))])
        return indices
    
    def max_neighbourhood_individual(self, coords, values):
        """
        Returns grid position of max value in neighborhood for a single agent.

        Args:
            coords (list): Neighborhood coordinates for one agent.
            values (list): Neighborhood values for one agent.

        Returns:
            tuple: Coordinate of max value.
        """
        indices = coords[values.index(max(values))]
        return indices

    def out_of_bounds(self, pos):
        """
        Determines whether a position is off the grid.

        Args:
            pos (tuple): (y, x) position.

        Returns:
            bool: True if out of bounds, False otherwise.
        """
        y, x = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def torus_adj(self, pos):
        """
        Convert coordinate, handling torus looping. Code by Mesa: https://github.com/projectmesa/mesa. 
        Args:
            pos (tuple): (y, x) position.

        Returns:
            tuple: Adjusted position.
        """
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            return  pos[1] % self.height, pos[0] % self.width 


#------------------------------------------------
#        SOCIAL NETWORK (NEIGHBOURS)
#------------------------------------------------
        
    def Distance(self):
        """
        Compute distance matrix for every farmer.

        Returns:
            tuple: (distance_matrix, index_neighbours)
        """
        distance_matrix = [] 

        index_neighbours = []
        for i in range(self.n):
            x_self = np.full(self.n,self.lonlat[i, 0]) 
            y_self = np.full(self.n, self.lonlat[i, 1])
            distance = np.sqrt(abs((x_self - self.lonlat[:, 0])**2 + (y_self - self.lonlat[:, 1])**2) )
            non_zeros = np.delete(distance, i) # do not include agent itself
            neighbourhood_distance = non_zeros[non_zeros <= self.radius_neighbourhood]
            distance_matrix.append(neighbourhood_distance)

            index_neighbours.append(np.delete(self.activation_order,i)[non_zeros <= self.radius_neighbourhood])

        return distance_matrix, index_neighbours #, x_neighbours, y_neighbours


    def Neighbours_adopted(self, measure):
        """
        Calculate the share of people in the neighborhood that have adopted a measure.

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Share of neighbors adopted for each agent.
        """

        share_neighbours_adopted = np.zeros(self.n)

        adoption_arrays = [
            self.adapt_measure_0,
            self.adapt_measure_1,
            self.adapt_measure_2,
            self.adapt_measure_3,
            self.adapt_measure_4,
            self.adapt_measure_5,
        ]
        adopted_m = adoption_arrays[measure]
 
        for i in range(self.n):
    
            adopted = adopted_m[self.index_neighbours[i]]
            nr_neighbours = len(self.index_neighbours[i])
            share_neighbours_adopted[i] = np.where(nr_neighbours > 0, np.sum(adopted) / nr_neighbours, 0)
            
        return  share_neighbours_adopted

    def Neighbours_adopted_attributes(self, measure, attribute, relative):
        """
        Calculate the average of an attribute among neighbors who adopted a measure.

        Args:
            measure (int): Adaptation measure index.
            attribute (np.ndarray): Attribute values.
            relative (np.ndarray): Normalization factor.

        Returns:
            np.ndarray: Average attribute among adopted neighbors.
        """
        
        average = np.zeros(self.n)

        # Select the correct adoption array
        adoption_arrays = [
            self.adapt_measure_0,
            self.adapt_measure_1,
            self.adapt_measure_2,
            self.adapt_measure_3,
            self.adapt_measure_4,
            self.adapt_measure_5,
        ]
        adopted = adoption_arrays[measure]

        for i in range(self.n):
                attribute_new = attribute[self.index_neighbours[i]]/relative[i] # make relative
                adopted_attributes = adopted[self.index_neighbours[i]] * attribute_new # multiply with adoption [0,1]
                if relative[i] == 0.0:
                    average[i] = 0.0
                else:
                    average[i]= adopted_attributes[adopted_attributes > 0].mean()
             
        average[np.isnan(average)] = 0
        average_neighbours_adopted = average
        assert (average_neighbours_adopted != np.nan).all()
        assert (average_neighbours_adopted >= 0).all()

        return  average_neighbours_adopted