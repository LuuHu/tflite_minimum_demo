

#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <vector>

struct BboxWithScore
{
    float tx, ty, bx, by, area, score; //（tx,ty）检测框左上角顶点坐标 ，  （bx,by） 检测框右下角顶点坐标 ， area 检测框面积 ，score 得分
};

void softNms(std::vector<BboxWithScore> &bboxes, const int &method, const float &sigma=0.5, const float &iou_thre=0.6, const float &threshold=0.1);

float calIOU_softNms(const BboxWithScore &bbox1, const BboxWithScore &bbox2);

#endif