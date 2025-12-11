//=============================================================================
// Copyright 2024 Astrid Bunge, Sven Wagner, Dennis Bukenberger, Mario Botsch, Marc Alexa
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#ifndef GLOBALENUMS_H
#define GLOBALENUMS_H

#include <string>

enum LaplaceMode2D
{
    PolySimpleLaplace = 0,
    AlexaWardetzkyLaplace = 1,
    Diamond = 2,
    deGoesLaplace = 3,
    Harmonic = 4,
};

enum InsertedPoint
{
    Centroid_ = 0,
    AreaMinimizer = 2,
    TraceMinimizer = 3,
};

enum DiffusionStep
{
    MeanEdge = 0,
    MaxEdge = 1,
    MaxDiagonal = 2
};

class SmoothingConfigs
{
public:
    explicit SmoothingConfigs(int numIters, bool fixBoundary = false,
                              bool updateQuadrics = false,
                              bool withCnum = false,
                              bool generalizedCnum = false,
                              bool lockFaceVirtuals = false)
        : numIters(numIters),
          fixBoundary(fixBoundary),
          updateQuadrics(updateQuadrics),
          withCnum(withCnum),
          generalizedCnum(generalizedCnum),
          lockFaceVirtuals(lockFaceVirtuals) {};

    int numIters;
    bool fixBoundary;
    bool updateQuadrics;
    bool withCnum;
    bool generalizedCnum;

    bool lockFaceVirtuals; // if true, do not recompute f:Virtuals inside PolySmoothing
    bool useSTO;           // optional semantic: mark that STO was used
    double sto_fraction;   
};

#endif
