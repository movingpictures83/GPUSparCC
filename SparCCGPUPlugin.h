#ifndef SPARCCGPUPLUGIN_H
#define SPARCCGPUPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>

class SparCCGPUPlugin : public Plugin
{
public: 
 std::string toString() {return "SparCCGPU";}
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
