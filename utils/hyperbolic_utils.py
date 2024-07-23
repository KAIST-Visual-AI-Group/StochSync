import torch
from Utils.extra_utils import ishomo, homo, unhomo
from Utils.matrix_utils import rodrigues
from math import pi

MAGIC_NUMBER = 0.485867977142334

NEARBY_KEYS = [
    "",
    "U", "D", "L", "R",

    "UU", "DD", "LL", "RR",
    "UL", "LU",
    "UR", "RU",
    "DL", "LD",
    "DR", "RD",

    "UUU", "DDD", "LLL", "RRR",
    "UUL", "ULU", "LUU", "LLU", "LUL", "ULL",
    "UUR", "URU", "RUU", "RRU", "RUR", "URR",
    "DDL", "DLD", "LDD", "LLD", "LDL", "DLL",
    "DDR", "DRD", "RDD", "RRD", "RDR", "DRR",

    "UUUU", "DDDD", "LLLL", "RRRR",
    "UUUL", "UULU", "ULUU", "LUUU", "LLLU", "LLUL", "LULL", "ULLL", "UULL", "ULUL", "ULLU", "LUUL", "LULU", "LLUU",
    "UUUR", "UURU", "URUU", "RUUU", "RRRU", "RRUR", "RURR", "URRR", "UURR", "URUR", "URRU", "RUUR", "RURU", "RRUU",
    "DDDL", "DDLD", "DLDD", "LDDD", "LLLD", "LLDL", "LDLL", "DLLL", "DDLL", "DLDL", "DLLD", "LDDL", "LDLD", "LLDD",
    "DDDR", "DDRD", "DRDD", "RDDD", "RRRD", "RRDR", "RDRR", "DRRR", "DDRR", "DRDR", "DRRD", "RDDR", "RDRD", "RRDD",

    "RUUL", "ULLD", "LDDR", "DRRU",
]
NEARBY_IDENTITY_RULES = {
    "<U":"L<",
    "<L":"D<",
    "<D":"R<",
    "<R":"U<",

    ">U":"R>",
    ">L":"U>",
    ">D":"L>",
    ">R":"D>",
}

IDENTITY_RULES = {
    "URRD": "RUUL>>",
    "LUUR": "ULLD>>",
    "DLLU": "LDDR>>",
    "RDDL": "DRRU>>",

    "LUR": "UL<",
    "URD": "RU<",
    "RDL": "DR<",
    "DLU": "LD<",
    "LDR": "DL>",
    "ULD": "LU>",
    "RUL": "UR>",
    "DRU": "RD>",
    "L<":"<U",
    "D<":"<L",
    "R<":"<D",
    "U<":"<R",

    "R>":">U",
    "U>":">L",
    "L>":">D",
    "D>":">R",
    
    "LR": "",
    "RL": "",
    "UD": "",
    "DU": "",

    "><": "",
    "<>": "",
}

def hyper_dot(a, b):
    # x1y1 + x2y2 + x3y3 - x4y4
    return torch.dot(a[:3], b[:3]) - a[3] * b[3]

def hyper_distance(a, b):
    a = homo(a)
    b = homo(b)

    ab = hyper_dot(a, b)
    aa = hyper_dot(a, a)
    bb = hyper_dot(b, b)
    d = 2 * torch.acosh(torch.sqrt((ab * ab) / (aa * bb)))
    return d

def hyper_reflect(a):
    # I - 2pp^T I^{3,1} / <p,p>
    I_31 = torch.diag(torch.tensor([1, 1, 1, -1], dtype=a.dtype, device=a.device))
    p = a.view(-1, 1)  # Ensure a is column vector
    pT = p.T
    numerator = 2 * p @ pT @ I_31
    denominator = hyper_dot(a, a)
    reflection_matrix = torch.eye(4, dtype=a.dtype, device=a.device) - numerator / denominator
    return reflection_matrix

def hyper_midopoint(a, b):
    # a*sqrt(<b,b><a,b>) + b*sqrt(<a,a><a,b>)
    dot_aa = hyper_dot(a, a)
    dot_bb = hyper_dot(b, b)
    dot_ab = hyper_dot(a, b)
    term1 = a * torch.sqrt(dot_bb * dot_ab)
    term2 = b * torch.sqrt(dot_aa * dot_ab)
    return term1 + term2

def hyper_translation(a, b):
    a = homo(a)
    b = homo(b)
    midpoint = hyper_midopoint(a, b)
    r_m = hyper_reflect(midpoint)
    r_a = hyper_reflect(a)
    return r_m @ r_a

def hyper_basis_shifting(basis, p1, p2):
    dtype = basis.dtype
    device = basis.device

    origin = torch.tensor([0,0,0,1], dtype=dtype, device=device)
    p1 = homo(p1).to(device)
    p2 = homo(p2).to(device)

    o_p1 = hyper_translation(origin, p1)
    p1_p2 = hyper_translation(p1, p2)
    p2_o = hyper_translation(p2, origin)
    adj_mtx = p2_o @ p1_p2 @ o_p1

    assert abs(torch.det(adj_mtx) - 1) < 1e-4, f"{torch.det(adj_mtx)} should be 1."
    adj_mtx = adj_mtx[:3, :3]
    
    return adj_mtx.inverse() @ basis

class HyperRectangle:
    @classmethod
    def remove_rotation(cls, key):
        while True:
            changed = False
            for idkey, replacement in NEARBY_IDENTITY_RULES.items():
                if idkey in key:
                    key = key.replace(idkey, replacement)
                    changed = True
                    break
            if not changed:
                break
        return key.replace("<", "").replace(">", "")
    
    @classmethod
    def condense_key(cls, key):
        # comdense the key by iteratively applying the identity rules
        while True:
            changed = False
            for idkey, replacement in IDENTITY_RULES.items():
                if idkey in key:
                    key = key.replace(idkey, replacement)
                    changed = True
                    break
            if not changed:
                break
        return key

    def __init__(self, a, b, c, d):
        self.key = ""
        self.init_a = a
        self.init_b = b
        self.init_c = c
        self.init_d = d
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.accum_matrix = torch.eye(4, dtype=a.dtype)
    
    def get_points(self):
        points = [
            self.a[:3],
            self.b[:3],
            self.c[:3],
            self.d[:3]
        ]
        points_stack = torch.stack(points)  # 4x3
        return points_stack
    
    def get_linestrip(self):
        points = [
            self.a[:3],
            self.b[:3],
            self.c[:3],
            self.d[:3],
            self.a[:3]
        ]
        points_stack = torch.stack(points)
        return points_stack
    
    def get_trianglelist(self):
        points = [
            self.a[:3],
            self.b[:3],
            self.c[:3],
            self.a[:3],
            self.c[:3],
            self.d[:3]
        ]
        points_stack = torch.stack(points)
        return points_stack
    
    def move_along_key(self, key):
        for d in key:
            if d == "U":
                self.move_up()
            elif d == "D":
                self.move_down()
            elif d == "L":
                self.move_left()
            elif d == "R":
                self.move_right()
            elif d == ">":
                self.rotate_right()
            elif d == "<":
                self.rotate_left()
    
    def condense(self):
        new_key = self.condense_key(self.key)
        self.key = ""
        self.a, self.b, self.c, self.d = self.init_a, self.init_b, self.init_c, self.init_d
        self.accum_matrix = torch.eye(4, dtype=self.a.dtype)
        self.move_along_key(new_key)
    
    def get_center(self):
        ab = hyper_midopoint(self.a, self.b)
        cd = hyper_midopoint(self.c, self.d)
        center = hyper_midopoint(ab, cd)
        return center / center[3].item()

    def get_t_right(self):
        ab = hyper_midopoint(self.a, self.b)
        cd = hyper_midopoint(self.c, self.d)

        t_right = hyper_translation(ab, cd)
        return t_right
    
    def get_t_up(self):
        bc = hyper_midopoint(self.b, self.c)
        ad = hyper_midopoint(self.a, self.d)

        t_up = hyper_translation(bc, ad)
        return t_up
    
    def get_t_left(self):
        cd = hyper_midopoint(self.c, self.d)
        ab = hyper_midopoint(self.a, self.b)

        t_left = hyper_translation(cd, ab)
        return t_left
    
    def get_t_down(self):
        ad = hyper_midopoint(self.a, self.d)
        bc = hyper_midopoint(self.b, self.c)

        t_down = hyper_translation(ad, bc)
        return t_down
    
    def move(self, matrix):
        self.a = matrix @ self.a
        self.b = matrix @ self.b
        self.c = matrix @ self.c
        self.d = matrix @ self.d

        self.a /= self.a[3].item()
        self.b /= self.b[3].item()
        self.c /= self.c[3].item()
        self.d /= self.d[3].item()
        
        self.accum_matrix = matrix @ self.accum_matrix
    
    def rotate_left(self):
        # a -> b -> c -> d -> a
        self.a, self.b, self.c, self.d = self.b, self.c, self.d, self.a
        rotation_matrix = torch.eye(4, dtype=torch.float)
        rotation_matrix[:3,:3] = rodrigues(torch.tensor([0, -1, 0], dtype=self.a.dtype), 90 * pi / 180)
        self.accum_matrix = rotation_matrix @ self.accum_matrix
        self.key += "<"

    def rotate_right(self):
        # a -> d -> c -> b -> a
        self.a, self.b, self.c, self.d = self.d, self.a, self.b, self.c
        rotation_matrix = torch.eye(4, dtype=torch.float)
        rotation_matrix[:3,:3] = rodrigues(torch.tensor([0, -1, 0], dtype=self.a.dtype), -90 * pi / 180)
        self.accum_matrix = rotation_matrix @ self.accum_matrix
        self.key += ">"

    def move_right(self):
        t_right = self.get_t_right()
        self.move(t_right)
        self.key += "R"

    def move_up(self):
        t_up = self.get_t_up()
        self.move(t_up)
        self.key += "U"
    
    def move_left(self):
        t_left = self.get_t_left()
        self.move(t_left)
        self.key += "L"

    def move_down(self):
        t_down = self.get_t_down()
        self.move(t_down)
        self.key += "D"
    
    def __str__(self):
        a_list = [round(x, 3) for x in self.a[:3].tolist()]
        b_list = [round(x, 3) for x in self.b[:3].tolist()]
        c_list = [round(x, 3) for x in self.c[:3].tolist()]
        d_list = [round(x, 3) for x in self.d[:3].tolist()]
        return f"a: {a_list}, b: {b_list}, c: {c_list}, d: {d_list}"
    
    @classmethod
    def at_center(self, r):
        a = torch.tensor([-r, 0, r, 1], dtype=torch.float)
        b = torch.tensor([-r, 0, -r, 1], dtype=torch.float)
        c = torch.tensor([r, 0, -r, 1], dtype=torch.float)
        d = torch.tensor([r, 0, r, 1], dtype=torch.float)
        return HyperRectangle(a, b, c, d)
    
    @classmethod
    def origin_left_t(self):
        tmp = HyperRectangle.at_center(MAGIC_NUMBER)
        left_t = tmp.get_t_left()
        return left_t
    
    @classmethod
    def origin_right_t(self):
        tmp = HyperRectangle.at_center(MAGIC_NUMBER)
        right_t = tmp.get_t_right()
        return right_t
    
    @classmethod
    def origin_up_t(self):
        tmp = HyperRectangle.at_center(MAGIC_NUMBER)
        up_t = tmp.get_t_up()
        return up_t
    
    @classmethod
    def origin_down_t(self):
        tmp = HyperRectangle.at_center(MAGIC_NUMBER)
        down_t = tmp.get_t_down()
        return down_t
    
    @classmethod
    def origin_t(self, key):
        tmp = HyperRectangle.at_center(MAGIC_NUMBER)
        tmp.move_along_key(key)
        return tmp.accum_matrix
    


# Test the hyperbolic translation
r1 = HyperRectangle.at_center(MAGIC_NUMBER)
r1.move_right()
r1.move_up()
a = r1.a

r2 = HyperRectangle.at_center(MAGIC_NUMBER)
r2.move_up()
r2.move_right()
c = r2.c

assert torch.allclose(a, c, atol=1e-6), f"Expected {a}=={c}"

