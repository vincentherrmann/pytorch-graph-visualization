import torch

# 0 [[0, 1, -1, 2]]

# 0 [[0, -1, 1, 2],
# 1  [-1, -1, 3, -1],
# 2  [-1, -1, -1, 4]]

# 0 [[0, 1, -1, -1],
# 1  [-1, 2, -1, -1],
# 2  [...],
# 3  [...],
# 4  [...]]


class QuadTree:
    def __init__(self, pos, mass, max_levels=100, device='cpu'):
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
            # remove unused indices

            q_com = torch.zeros(num_sections*4, 2, device=self.device)
            q_com[:, 0].scatter_add_(0, sections, pos[:, 0])
            q_com[:, 1].scatter_add_(0, sections, pos[:, 1])
            q_com /= q_mass.unsqueeze(1)

            continued_quadrants = q_mass > 0
            non_empty_q = continued_quadrants.nonzero().squeeze(1)
            new_indices = torch.arange(non_empty_q.shape[0], device=self.device)
            section_indexing = torch.zeros(num_sections, 4, dtype=torch.long, device=self.device) - 1.
            section_indexing[non_empty_q / 4, non_empty_q % 4] = new_indices
            num_sections = non_empty_q.shape[0]

            print("section_indexing_shape:", section_indexing.shape)

            q_mass = q_mass[continued_quadrants]
            q_com = q_com[continued_quadrants, :]
            self.quadrant_mass.append(q_mass)
            self.center_of_mass.append(q_com)
            self.section_indexing.append(section_indexing)

            sections = section_indexing[sections / 4, sections % 4]

            # discard points that already have their own section
            continued_points = torch.gt(q_mass[sections], mass)
            if torch.sum(continued_points) < 1:
                break
            if self.levels >= self.max_levels:
                print("too many levels!")
            pos = pos[continued_points]
            mass = mass[continued_points]
            sections = sections[continued_points]
            norm_pos = norm_pos[continued_points]
            print("max mass:", torch.max(q_mass))

        pass

    def traverse(self, x, m, mac=0.7, gravity=-0.05):
        force = torch.zeros_like(x)
        pairs = torch.cat([torch.arange(x.shape[0], dtype=torch.long, device=self.device).unsqueeze(1).repeat(1, 4).view(-1, 1),
                           torch.arange(4, dtype=torch.long, device=self.device).unsqueeze(1).repeat(x.shape[0], 1)], dim=1)
        for l in range(self.levels):
            #print("pairs:", pairs.shape[0])
            indexing = self.section_indexing[l]
            pairs[:, 1] = indexing[pairs[:, 1] / 4, pairs[:, 1] % 4]
            pairs = pairs[pairs[:, 1] > 0, :]

            this_com = self.center_of_mass[l][pairs[:, 1], :]
            this_mass = self.quadrant_mass[l][pairs[:, 1]]

            diff = x[pairs[:, 0], :] - this_com
            dist = torch.norm(diff, 2, dim=1)
            section_size = self.size / 2 ** (l + 1)
            d2r = section_size / dist

            accept = torch.le(d2r, mac) + torch.isnan(d2r)
            #print("accepted:", torch.sum(accept).item(), "- ", 100*torch.sum(accept).item() / pairs.shape[0], '%')

            this_f = gravity * (this_mass[accept] * m[pairs[:, 0]][accept] / dist[accept]**3).unsqueeze(1) * diff[accept]
            force[:, 0].scatter_add_(0, pairs[:, 0][accept], this_f[:, 0])
            force[:, 1].scatter_add_(0, pairs[:, 0][accept], this_f[:, 1])

            refine = pairs[(accept == 0).nonzero(), :].squeeze(1)
            #refine[:, 1] = indexing[refine[:, 1] / 4, refine[:, 1] % 4]
            refine[:, 1] *= 4
            refine = refine.unsqueeze(1).repeat(1, 4, 1)
            refine[:, :, 1] = refine[:, :, 1] + torch.LongTensor([0, 1, 2, 3], device=self.device).unsqueeze(0)
            pairs = refine.view(-1, 2)

            pass
        pass





