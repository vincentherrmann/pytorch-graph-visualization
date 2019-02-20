import torch


def gravity_function(m1, m2, difference, distance):
    return -(m1 * m2 / distance**3).unsqueeze(1) * difference

def energy_function(m1, m2, difference, distance):
    return -(m1 * m2 / distance**3).unsqueeze(1).repeat(1, 3)

# 0 [[0, 1, -1, 2]]

# 0 [[0, -1, 1, 2],
# 1  [-1, -1, 3, -1],
# 2  [-1, -1, -1, 4]]

# 0 [[0, 1, -1, -1],
# 1  [-1, 2, -1, -1],
# 2  [...],
# 3  [...],
# 4  [...]]


class QuadTree(object):
    def __init__(self, pos, mass, max_levels=100, device='cpu'):
        super().__init__()
        self.levels = 0
        self.max_levels = max_levels
        self.device = device

        min_val = torch.min(pos) - 1e-4
        max_val = torch.max(pos) + 1e-4
        self.size = max_val - min_val

        norm_pos = torch.zeros_like(pos)
        norm_pos[:, 0] = (pos[:, 0] - min_val) / self.size
        norm_pos[:, 1] = (pos[:, 1] - min_val) / self.size

        self.quadrant_mass = []
        self.center_of_mass = []
        self.section_indexing = []
        self.is_end_node = []
        sections = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)
        num_sections = 1

        while True:
            self.levels += 1
            num_divisions = 2**self.levels

            # calculate the section in which each point falls
            point_quadrant = torch.floor(norm_pos * num_divisions).long()
            point_quadrant = point_quadrant[:, 0] % 2 + 2 * (point_quadrant[:, 1] % 2)
            sections *= 4
            sections += point_quadrant

            # calculate total mass and center of mass for each section
            q_mass = torch.zeros(num_sections*4, device=self.device)
            q_mass.scatter_add_(0, sections, mass)

            q_com = torch.zeros(num_sections*4, 2, device=self.device)
            q_com[:, 0].scatter_add_(0, sections, pos[:, 0])
            q_com[:, 1].scatter_add_(0, sections, pos[:, 1])
            q_com /= q_mass.unsqueeze(1)

            continued_points = q_mass[sections] > mass
            end_nodes = sections[continued_points == 0]

            empty_node = q_mass == 0.
            #empty_node[end_nodes] = 1
            continued_quadrants = 1 - empty_node
            non_empty_q = continued_quadrants.nonzero().squeeze(1)
            new_indices = torch.arange(non_empty_q.shape[0], device=self.device)
            section_indexing = torch.zeros(num_sections, 4, dtype=torch.long, device=self.device) - 1.
            section_indexing[non_empty_q / 4, non_empty_q % 4] = new_indices
            num_sections = non_empty_q.shape[0]

            #print("section_indexing_shape:", section_indexing.shape)

            q_end = torch.zeros_like(continued_quadrants)
            q_end[end_nodes] = 1
            q_end = q_end[continued_quadrants]
            q_mass = q_mass[continued_quadrants]
            q_com = q_com[continued_quadrants, :]
            self.quadrant_mass.append(q_mass)
            self.center_of_mass.append(q_com)
            self.section_indexing.append(section_indexing)
            self.is_end_node.append(q_end)

            sections = section_indexing[sections / 4, sections % 4]

            # discard points that already have their own section
            if torch.sum(continued_points) < 1:
                break
            if self.levels >= self.max_levels:
                print("too many levels!")
                break
            pos = pos[continued_points]
            mass = mass[continued_points]
            sections = sections[continued_points]
            norm_pos = norm_pos[continued_points]
            #print("max mass:", torch.max(q_mass))

    def traverse(self, x, m, mac=0.7, gravity=-0.05, force_function=gravity_function):
        #print("levels:", self.levels)
        force = torch.zeros_like(x)
        pairs_o = torch.cat([torch.arange(x.shape[0], dtype=torch.long, device=self.device).unsqueeze(1).repeat(1, 4).view(-1, 1),
                           torch.arange(4, dtype=torch.long, device=self.device).unsqueeze(1).repeat(x.shape[0], 1)], dim=1)
        for l in range(self.levels):
            indexing = self.section_indexing[l]
            pairs = pairs_o.clone()
            pairs[:, 1] = indexing[pairs_o[:, 1] / 4, pairs_o[:, 1] % 4]
            pairs = pairs[pairs[:, 1] >= 0, :]
            #print("pairs shape:", pairs.shape)

            this_com = self.center_of_mass[l][pairs[:, 1], :]
            this_mass = self.quadrant_mass[l][pairs[:, 1]]

            diff = x[pairs[:, 0], :] - this_com
            dist = torch.norm(diff, 2, dim=1)
            section_size = self.size / 2 ** (l + 1)
            d2r = section_size / dist

            #if l < self.levels - 1:
            accept = d2r < mac
            end_node = self.is_end_node[l][pairs[:, 1]]
            accept = torch.max(accept, end_node) - (dist <= 1e-9)
            #else:
            #    accept = 1 - torch.isnan(d2r)

            #print("num accepted:", torch.sum(accept).item())

            this_f = force_function(m1=this_mass[accept],
                                    m2=m[pairs[:, 0]][accept],
                                    difference=diff[accept],
                                    distance=dist[accept])
            #this_f = -gravity * (this_mass[accept] * m[pairs[:, 0]][accept] / (dist[accept]**3 + 1e-5)).unsqueeze(1) * diff[accept]
            force[:, 0].scatter_add_(0, pairs[:, 0][accept], this_f[:, 0])
            force[:, 1].scatter_add_(0, pairs[:, 0][accept], this_f[:, 1])

            refine = pairs[(accept == 0).nonzero(), :].squeeze(1)
            refine[:, 1] *= 4
            refine = refine.unsqueeze(1).repeat(1, 4, 1)
            refine[:, :, 1] = refine[:, :, 1] + torch.arange(4, dtype=torch.long, device=self.device).unsqueeze(0)
            pairs_o = refine.view(-1, 2)

        return force









