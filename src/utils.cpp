
#include "utils.hpp"
#include <algorithm>
#include <cmath>

void softNms(std::vector<BboxWithScore> &bboxes, const int &method, const float &sigma, const float &iou_thre, const float &threshold)
{
    if (bboxes.empty())
    {
        return;
    }

    int N = bboxes.size();
    float max_score, max_pos, cur_pos, weight;
    BboxWithScore tmp_bbox, index_bbox;
    for (int i = 0; i < N; ++i)
    {
        max_score = bboxes[i].score;
        max_pos = i;
        tmp_bbox = bboxes[i];
        cur_pos = i + 1;

        while (cur_pos < N)
        {
            if (max_score < bboxes[cur_pos].score)
            {
                max_score = bboxes[cur_pos].score;
                max_pos = cur_pos;
            }
            cur_pos++;
        }

        bboxes[i] = bboxes[max_pos];

        bboxes[max_pos] = tmp_bbox;
        tmp_bbox = bboxes[i];

        cur_pos = i + 1;

        while (cur_pos < N)
        {
            index_bbox = bboxes[cur_pos];

            float area = index_bbox.bx * index_bbox.by;
            float iou = calIOU_softNms(tmp_bbox, index_bbox);
            if (iou <= 0)
            {
                cur_pos++;
                continue;
            }
            iou /= area;
            if (method == 1)
            {
                if (iou > iou_thre)
                {
                    weight = 1 - iou;
                }
                else
                {
                    weight = 1;
                }
            }
            else if (method == 2)
            {
                weight = exp(-(iou * iou) / sigma);
            }
            else // original NMS
            {
                if (iou > iou_thre)
                {
                    weight = 0;
                }
                else
                {
                    weight = 1;
                }
            }
            bboxes[cur_pos].score *= weight;
            if (bboxes[cur_pos].score <= threshold)
            {
                bboxes[cur_pos] = bboxes[N - 1];
                N--;
                cur_pos = cur_pos - 1;
            }
            cur_pos++;
        }
    }

    bboxes.resize(N);
}

float calIOU_softNms(const BboxWithScore &bbox1, const BboxWithScore &bbox2)
{
    float iw = (std::min(bbox1.tx + bbox1.bx / 2., bbox2.tx + bbox2.bx / 2.) -
                std::max(bbox1.tx - bbox1.bx / 2., bbox2.tx - bbox2.bx / 2.));
    if (iw < 0)
    {
        return 0.;
    }

    float ih = (std::min(bbox1.ty + bbox1.by / 2., bbox2.ty + bbox2.by / 2.) -
                std::max(bbox1.ty - bbox1.by / 2., bbox2.ty - bbox2.by / 2.));

    if (ih < 0)
    {
        return 0.;
    }

    return iw * ih;
}
