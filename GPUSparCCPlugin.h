#ifndef GPUSPARCCPLUGIN_H
#define GPUSPARCCPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>

class GPUSparCCPlugin : public Plugin
{
public: 
 std::string toString() {return "GPUSparCC";}
 void input(std::string file);
 void run();
 void output(std::string file);

private: 
 std::string inputfile;
 std::string outputfile;
 std::vector<std::string> taxa;
 int rows;
 int cols;
 double* a;
 double* b;
};

#endif
