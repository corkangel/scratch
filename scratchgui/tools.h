
#pragma once


class sModel;
class sLearner;

void DrawTensorLogs();
void DrawTensorTable();
void DrawModel(const sLearner& learner, const sModel& model);
void DrawActivationStats(const sLearner& learner, const sModel& model);


using initFunc = void(*)();
bool DrawMenu(sLearner& learner, initFunc init);