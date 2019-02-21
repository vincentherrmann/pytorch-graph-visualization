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


class BarnesHutTree(object):
    def __init__(self, pos, mass, max_levels=100, device='cpu'):
        super().__init__()
        self.levels = 0
        self.max_levels = max_levels
        self.device = device

        self.num_dim = pos.shape[1]
        self.num_o = 2**self.num_dim

        min_val = torch.min(pos) - 1e-4
        max_val = torch.max(pos) + 1e-4
        self.size = max_val - min_val

        norm_pos = (pos - min_val.unsqueeze(0)) / self.size.unsqueeze(0)

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
            point_orthant = torch.floor(norm_pos * num_divisions).long()
            point_orthant = torch.sum((point_orthant % 2) * (2**torch.arange(self.num_dim, device=self.device)).unsqueeze(0), dim=1)

            sections *= self.num_o
            sections += point_orthant

            # calculate total mass and center of mass for each section
            q_mass = torch.zeros(num_sections * self.num_o, device=self.device)
            q_mass.scatter_add_(0, sections, mass)

            q_com = torch.zeros(num_sections * self.num_o, self.num_dim, device=self.device)
            for d in range(self.num_dim):
                q_com[:, d].scatter_add_(0, sections, pos[:, d])
            q_com /= q_mass.unsqueeze(1)

            continued_points = q_mass[sections] > mass
            end_nodes = sections[continued_points == 0]

            empty_node = q_mass == 0.
            #empty_node[end_nodes] = 1
            continued_quadrants = 1 - empty_node
            non_empty_q = continued_quadrants.nonzero().squeeze(1)
            new_indices = torch.arange(non_empty_q.shape[0], device=self.device)
            section_indexing = torch.zeros(num_sections, self.num_o, dtype=torch.long, device=self.device) - 1.
            section_indexing[non_empty_q / self.num_o, non_empty_q % self.num_o] = new_indices
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

            sections = section_indexing[sections / self.num_o, sections % self.num_o]

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

    def traverse(self, x, m, mac=0.7, force_function=gravity_function):
        #print("levels:", self.levels)
        force = torch.zeros_like(x)
        pairs_o = torch.cat([torch.arange(x.shape[0], dtype=torch.long, device=self.device).unsqueeze(1).repeat(1, self.num_o).view(-1, 1),
                           torch.arange(self.num_o, dtype=torch.long, device=self.device).unsqueeze(1).repeat(x.shape[0], 1)], dim=1)
        for l in range(self.levels):
            indexing = self.section_indexing[l]
            pairs = pairs_o.clone()
            pairs[:, 1] = indexing[pairs_o[:, 1] / self.num_o, pairs_o[:, 1] % self.num_o]
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
            refine[:, 1] *= self.num_o
            refine = refine.unsqueeze(1).repeat(1, self.num_o, 1)
            refine[:, :, 1] = refine[:, :, 1] + torch.arange(self.num_o, dtype=torch.long, device=self.device).unsqueeze(0)
            pairs_o = refine.view(-1, 2)

        return force









