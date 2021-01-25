'''This module defines classes PlaneEquation and CutPlane'''
import numpy as np
from neurom import COLS
from pyquaternion import Quaternion

A, B, C, D = range(4)
ABC = slice(0, 3)


def _get_displaced_pos(pos, quat, size_multiplier, axis):
    """ Compute the shifted position wrt the quaternion and axis """
    return pos + size_multiplier * np.array(quat.rotate(axis))


class PlaneEquation(object):
    '''This class defines the equation of a plane.
    It is a mathematical object which is domain agnostic.

    a X + b Y + c Z + d = 0
    '''

    def __init__(self, a, b, c, d):
        if (a, b, c) == (0, 0, 0):
            raise ValueError('Cannot define a plane with a == b == c == 0')
        self.coefs = np.array([a, b, c, d], dtype=np.float)

    @classmethod
    def from_dict(cls, data):
        '''Instantiate object from dict like:
             {"a": 1, "b": 2, "c": 0, "d": 10}
        '''
        return cls(data['a'], data['b'], data['c'], data['d'])

    @classmethod
    def from_rotations_translations(cls, transformations):
        '''Factory method to build the plane equation from the rotation and translation
        provided by the cut_plane.viewer

        Args:
            transformations: An array [rot_x, rot_y, rot_z, transl_x, transl_y, transl_z]
        '''
        rot_x, rot_y, rot_z, transl_x, transl_y, transl_z = transformations
        qx = Quaternion(axis=[1, 0, 0], angle=rot_x / 180. * np.pi).unit
        qy = Quaternion(axis=[0, 1, 0], angle=rot_y / 180. * np.pi).unit
        qz = Quaternion(axis=[0, 0, 1], angle=rot_z / 180. * np.pi).unit
        quat = qx * qy * qz
        normal_vector = _get_displaced_pos([0, 0, 0], quat, 100, (0, 0, 1))
        transl = np.array([transl_x, transl_y, transl_z])
        return cls(normal_vector[0],
                   normal_vector[1],
                   normal_vector[2],
                   -transl.dot(normal_vector))

    def distance(self, points):
        '''Returns an array containing the distance between the plane and each point'''
        points = np.array(points)
        u = self.coefs[ABC].copy()
        return np.abs(points.dot(u) + self.coefs[D]) / np.linalg.norm(u)

    def __str__(self):
        return '({0}) * X + ({1}) * Y + ({2}) * Z + ({3}) = 0'.format(*self.coefs)

    def to_json(self):
        '''Returns a dict for json serialization'''
        return dict(a=self.coefs[A], b=self.coefs[B], c=self.coefs[C], d=self.coefs[D],
                    comment="Equation: a*X + b*Y + c*Z + d = 0")

    @property
    def normal(self):
        '''Returns the vector orthogonal to the plane'''
        return self.coefs[ABC] / np.linalg.norm(self.coefs[ABC])

    def project_on_normal(self, points):
        '''Returns the points projected on the normal vector'''
        if self.coefs[A]:
            point_in_the_plane = np.array([-self.coefs[D] / self.coefs[A], 0, 0])
        elif self.coefs[B]:
            point_in_the_plane = np.array([0, -self.coefs[D] / self.coefs[B], 0])
        else:  # __init__ would have raised if c was 0 as well
            point_in_the_plane = np.array([0, 0, -self.coefs[D] / self.coefs[C]])

        return points[:, COLS.XYZ].dot(self.normal) - point_in_the_plane.dot(self.normal)

    def count_near_plane(self, points, bin_width):
        '''Return the number of points in ]-bin_width, 0] and ]0, bin_width]'''
        points = self.project_on_normal(points)
        n_left = len(points[(points > -bin_width) & (points <= 0)])
        n_right = len(points[(points > 0) & (points <= bin_width)])
        return n_left, n_right


class HalfSpace(PlaneEquation):
    '''
    A mathematical half-space: https://en.wikipedia.org/wiki/Half-space_(geometry)

    a X + b Y + c Z + d > 0 if upward == True
    a X + b Y + c Z + d < 0 else
    '''

    def __init__(self, a, b, c, d, upward):
        super().__init__(a, b, c, d)
        self.upward = upward

    def to_json(self):
        '''Returns a dict for json serialization'''
        return dict(a=self.coefs[A], b=self.coefs[B], c=self.coefs[C], d=self.coefs[D],
                    upward=self.upward,
                    comment="Equation: a*X + b*Y + c*Z + d {} 0".format(
                        '>' if self.upward else '<'))

    def project_on_directed_normal(self, points):
        '''Project on the normal oriented toward the inside of the half-space'''
        points = self.project_on_normal(points)
        if self.upward:
            return points
        else:
            return -points
