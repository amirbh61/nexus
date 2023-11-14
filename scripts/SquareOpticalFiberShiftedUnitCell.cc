#include "SquareOpticalFiber.hh"
#include "G4VisAttributes.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "OpticalMaterialProperties.h"
#include "MaterialsList.h"
#include "FactoryBase.h"
#include "G4SubtractionSolid.hh"
#include "G4VSolid.hh"
#include "G4LogicalBorderSurface.hh"
#include <vector>
#include "G4VSensitiveDetector.hh"
#include <iomanip>
#include "CLHEP/Random/Random.h"
#include <optional>
#include "SquareFiberSD.h"
#include "G4SDManager.hh"



/// To DO, 22.5.23:
// doesn't write to TPB file -> DONE
// make sure TPB is fixed on nexus -> might be ok

// ask gonzalo to write a new PersistencyManager that doesn't save the hd5 - DONE, changed .init settings
// ask Lior the length of fibers, min is best to avoid losing light 

// "mpt->AddConstProperty("WLSMEANNUMBERPHOTONS", num); is causing effect. It seems that G4's OpWLS
// is only for the absorption of UV photon and has nothing to do with the emissions. larger 
// values of WLSMEANNUMBERPHOTONS cause more WLS photons -> DONE

// To DO, 19.5.23:
// writing output file to /dev/null doesn't work now - stream of warning messages
// for 462K photons .h5 file size of ~500mb without SaveAllSteppingAction




namespace nexus{

REGISTER_CLASS(SquareOpticalFiber, GeometryBase)

SquareOpticalFiber::SquareOpticalFiber() : GeometryBase(), gen_(z_dist_seed_)
{  
msg_ = new G4GenericMessenger(this, "/Geometry/SquareOpticalFiber/", "Control commands of geometry SquareOpticalFiber."); 
msg_->DeclarePropertyWithUnit("specific_vertex", "mm",  specific_vertex_, "Set generation vertex.");
msg_->DeclarePropertyWithUnit("ELGap", "mm",  ELGap_, "Set EL gap size.");
msg_->DeclarePropertyWithUnit("pitch", "mm",  pitch_, "Set pitch/spacing size.");
msg_->DeclarePropertyWithUnit("distanceFiberHolder", "mm",  distanceFiberHolder_, "Set depth of fiber in holder.");
msg_->DeclarePropertyWithUnit("distanceAnodeHolder", "mm",  distanceAnodeHolder_, "Set distance between anode and holder.");
msg_->DeclarePropertyWithUnit("holderThickness", "mm",  holderThickness_, "Teflon holder thickness.");
msg_->DeclarePropertyWithUnit("TPBThickness", "um",  TPBThickness_, "TPB thickness.");

msgSD_ = new G4GenericMessenger(this, "/SquareOpticalFiber/SquareFiberSD/Output/",
                               "Commands for setting output file paths.");
msgSD_->DeclareProperty("sipmPath", sipmOutputFile_);
msgSD_->DeclareProperty("tpbPath", tpbOutputFile_);
}


SquareOpticalFiber::~SquareOpticalFiber(){
}



void SquareOpticalFiber::Construct(){
    G4int n = 25;
    bool claddingExists = true; // a flag -> needed for specific geometry change for cladding
    bool wallsExists = false; // a flag -> needed for specific geometry change for walls
    bool holderExists = true; // a flag -> needed for specific geometry change for holder
    bool holderTPBExist = true; // a flag -> needed for specific geometry change for holder TPB

  
    ///// Materials /////

    G4NistManager *nist = G4NistManager::Instance();
    G4double temperature = 298.*kelvin;
    G4double pressure = 10.*atmosphere;
    G4int sc_yield = 1; //value irrelevant for my usecase
    G4double e_lifetime = 1; //value irrelevant for my usecase

    G4Material *Xe = nist->FindOrBuildMaterial("G4_Xe", temperature, pressure);
    G4Material *Teflon = nist->FindOrBuildMaterial("G4_TEFLON", temperature, pressure); //detector barrel
    G4Material *Si = nist->FindOrBuildMaterial("G4_Si", temperature, pressure); //SiPM 
    G4Material *PMMA = materials::PMMA();
    G4Material *TPB = materials::TPB();
    G4Material *FPethylene = materials::FPethylene();

    // Optical Materials 
    Xe->SetMaterialPropertiesTable(opticalprops::GXe(pressure, temperature, sc_yield, e_lifetime ));
    Teflon->SetMaterialPropertiesTable(opticalprops::PTFE());
    PMMA->SetMaterialPropertiesTable(opticalprops::PMMA());
    TPB->SetMaterialPropertiesTable(opticalprops::TPB());
    Si->SetMaterialPropertiesTable(opticalprops::Si());
    FPethylene->SetMaterialPropertiesTable(opticalprops::FPethylene());

    // Optical surfaces - The same as in Nexus
    G4OpticalSurface* teflonSurface = new G4OpticalSurface("Teflon_Surface", unified,
                                                            ground, dielectric_metal);
    teflonSurface->SetMaterialPropertiesTable(opticalprops::PTFE());

    G4OpticalSurface* TPBFiberSurface = new G4OpticalSurface("TPB_surface", glisur,
                                                        ground, dielectric_dielectric, 0.01);
    TPBFiberSurface->SetMaterialPropertiesTable(opticalprops::TPB());

    G4OpticalSurface* VikuitiCoating = new G4OpticalSurface("Vikuiti_Surface", unified,
                                                                     polished, dielectric_metal);
    VikuitiCoating->SetMaterialPropertiesTable(opticalprops::Vikuiti());





    ///// Create the world /////
    G4double edge = 100*cm;

    G4Box *solidWorld = new G4Box("World", //name
                              edge, //side a
                              edge, //side b
                              edge); //side c

    G4LogicalVolume *worldLogicalVolume = new G4LogicalVolume(solidWorld, //the cylinder object
                                                                Xe, //material of cylinder
                                                                "World"); //name
    this->SetLogicalVolume(worldLogicalVolume);

    G4VPhysicalVolume* world = new G4PVPlacement(
                0,                // no rotation
                G4ThreeVector(0.,0.,0.),  // at (0,0,0)
                worldLogicalVolume, // its logical volume
                "World",          // its name
                0,                // its mother  volume
                false,            // no boolean operation
                0,                // copy number
                true);          // checking overlaps




    ///// Create the detector cylinder /////
    G4float delta = 1*mm; // SiPM is X*mm outside Cap1
    G4double BarrelThickness = 5*cm;
    G4double barrelOuterRadius = 50.*cm;
    G4double barrelInnerRadius = barrelOuterRadius - BarrelThickness;
    G4double cylLength = 7*cm + delta;

    G4Tubs *barrel = new G4Tubs("Barrel", //name
                              barrelInnerRadius, //inner radius
                              barrelOuterRadius, //outer radius
                              cylLength, //length
                              0., //initial revolve angle
                              2*M_PI); //final revolve angle

    G4LogicalVolume *barrelLogicalVolume = new G4LogicalVolume(
                                                                barrel, //the cylinder object
                                                                Teflon, //material of cylinder
                                                                "Barrel"); //name

    // new G4PVPlacement(
    //                 0,                // no rotation
    //                 G4ThreeVector(0.,0.,0.),// at (0,0,0)
    //                 barrelLogicalVolume, // its logical volume
    //                 "Barrel",          // its name
    //                 worldLogicalVolume, // its mother  volume
    //                 false,            // no boolean operation
    //                 0,                // copy number
    //                 true);          // checking overlaps


    // //make barrel surface reflective
    // new G4LogicalSkinSurface( "Teflon_coating", barrelLogicalVolume, teflonSurface);



    //create cap
    G4double capInnerRadius = 0*cm;
    G4double capOuterRadius = barrelOuterRadius;
    G4double capWidth = 2.5*cm; //half width
    G4double capLocationZ = cylLength+capWidth;
    G4Tubs *cap = new G4Tubs("Cap1", //name
                            capInnerRadius, //inner radius
                             capOuterRadius, //outer radius
                             capWidth, //length
                             0., //initial revolve angle
                             2*M_PI); //final revolve angle

    G4LogicalVolume *capLogicalVolume = new G4LogicalVolume(cap, //the cylinder object
                                                            Teflon, //material of cylinder
                                                            "Cap1"); //name

    new G4PVPlacement(
                    0,                // no rotation
                    G4ThreeVector(0.,0.,capLocationZ),
                    capLogicalVolume, // its logical volume
                    "Cap1",          // its name
                    worldLogicalVolume, // its mother  volume
                    false,            // no boolean operation
                    0,                // copy number
                    true);          // checking overlaps

    //add reflective surface to cap
    new G4LogicalSkinSurface( "Teflon_coating", capLogicalVolume, teflonSurface);


    ///// SIPM ARRAY /////
    G4double sideSiPM = 1.5*mm;

    G4double thicknessSiPM = sideSiPM/2;
    G4Box *SiPMSolidVolume = new G4Box("SiPM", //name
                              sideSiPM, //side a
                              sideSiPM, //side b
                              thicknessSiPM); //side c

    G4LogicalVolume* SiPMLogicalVolume = new G4LogicalVolume(SiPMSolidVolume, 
                                            Si, //material of cylinder
                                            "SiPM"); //name

 
    G4Color cyan = G4Color::Cyan();
    G4VisAttributes* colorSiPM = new G4VisAttributes(cyan);
    SiPMLogicalVolume->SetVisAttributes(colorSiPM);

    G4int numberOfSiPMs = n; //in a row/column , needs to be an ODD number!

    if (numberOfSiPMs%2 == 0) throw std::runtime_error("SiPM Number must be an ODD number !!");
    G4double maxCoord = (numberOfSiPMs-1) * pitch_ / 2.0;
    G4float zSiPM = cylLength + thicknessSiPM - delta;
    G4double halfPitch = pitch_ / 2.0;


    ///// SQUARE FIBER CORE /////
    G4double fiberSize = 1.5*mm;
    G4double holderWidth = 0.5*holderThickness_;
    G4double fiberLength = 0.5*(cylLength-delta);

    G4Box *fiberCore = new G4Box("Fiber_Core", //name
                           fiberSize, //side a
                           fiberSize, //side b
                           fiberLength); //length

    G4LogicalVolume *fiberCoreLogicalVolume = new G4LogicalVolume(
                                                        fiberCore, //the fiber object
                                                        // Polyethylene, //material of fiber
                                                        PMMA,
                                                        "Fiber_Core"); //name
    // new G4LogicalSkinSurface("fiberSurface", fiberCoreLogicalVolume, VikuitiCoatingTeflon);
    G4Color white(1,1,1);
    G4VisAttributes* fiberCoreColor = new G4VisAttributes(white);
    fiberCoreLogicalVolume->SetVisAttributes(fiberCoreColor);




    ///// SQUARE FIBER CLADDING ///// 
    G4double claddingSizeOuter = fiberSize + 0.01*fiberSize;
    G4double claddingSizeInner = fiberSize;
    G4double claddingLength = fiberLength;

    G4Box *fiberCladdingOuter = new G4Box("Fiber_Cladding_Outer",
                                            claddingSizeOuter,
                                            claddingSizeOuter,
                                            claddingLength);

    G4Box *fiberCladdingInner = new G4Box("Fiber_Cladding_Outer",
                                        claddingSizeInner,
                                        claddingSizeInner,
                                        claddingLength);

    G4SubtractionSolid *fiberCladding = new G4SubtractionSolid("Fiber_Cladding", fiberCladdingOuter, fiberCladdingInner);
    G4LogicalVolume *fiberCladdingLogicalVolume = new G4LogicalVolume(fiberCladding, FPethylene, "Fiber_Cladding");


    G4Color brown = G4Color::Brown();
    G4VisAttributes* claddingColor = new G4VisAttributes(brown);
    fiberCladdingLogicalVolume->SetVisAttributes(claddingColor);



    ///// TEFLON WALL SURROUNDING FIBER CORE /////
    G4double WallThickness = 0.5*mm;
    G4double outerTeflonWallSide = 0.5*pitch_;
    G4double innerTeflonWallSide = outerTeflonWallSide - WallThickness;
    G4double teflonWallLength = 0.5*(cylLength - distanceFiberHolder_ - holderWidth);
    // G4double teflonWallLength = 0.5*(cylLength - distanceFiberHolder_); //for a single fiber heatmap simulation


    G4Box *teflonWallOuter = new G4Box("Teflon_Wall_Outer", //name
                           outerTeflonWallSide, //side a 
                           outerTeflonWallSide, //side b 
                           teflonWallLength); //length

    G4Box *teflonWallInner = new G4Box("Teflon_wall_Inner",
                            innerTeflonWallSide,
                            innerTeflonWallSide,
                            teflonWallLength);

    G4SubtractionSolid *teflonWall = new G4SubtractionSolid("Teflon_Wall",teflonWallOuter,teflonWallInner); 

    G4LogicalVolume *teflonWallLogicalVolume = new G4LogicalVolume(teflonWall, Teflon, "Teflon_Wall");  

    //add reflective properties to teflon wall
    new G4LogicalSkinSurface( "Teflon_Wall", teflonWallLogicalVolume, teflonSurface);

    G4Color grey = G4Color::Grey();
    G4VisAttributes* teflonWallColor = new G4VisAttributes(grey);
    teflonWallLogicalVolume->SetVisAttributes(teflonWallColor);



    ///// TPB COATING ON FIBER END /////
    G4double TPBFiberSize = fiberSize;
    G4double TPBFiberWidth = 0.5*TPBThickness_;
    G4Box *fiberTPB = new G4Box("TPB_Fiber", //name
                           TPBFiberSize, //side a
                           TPBFiberSize, //side b
                           TPBFiberWidth); //length

    G4LogicalVolume *fiberTPBLogicalVolume = new G4LogicalVolume(
                                                        fiberTPB, //the fiber object
                                                        // Polyethylene, //material of fiber
                                                        TPB,
                                                        "TPB_Fiber"); //name

    G4Color green = G4Color::Green();
    G4VisAttributes* colorTPB = new G4VisAttributes(green);
    fiberTPBLogicalVolume->SetVisAttributes(colorTPB);


    ///// SQUARE LATTICE : CREATE PHYICAL VOLUMES FOR FIBER CORE, TEFLON WALL, SIPM, CLADDING  /////

    G4int numOfFibers = numberOfSiPMs;
    G4double zFiber = fiberLength;
    G4double zTeflonWall = teflonWallLength + holderWidth + distanceFiberHolder_;
    // G4double zTeflonWall = teflonWallLength + distanceFiberHolder_; //for a single fiber heatmap simulation
    G4double zFiberTPB = zFiber-fiberLength-TPBFiberWidth;

    std::vector<std::vector<double>> lattice_points;
    G4VPhysicalVolume *fiberTPBPhysicalVolume;
    G4VPhysicalVolume *fiberCorePhysicalVolume;
    G4VPhysicalVolume *fiberCladdingPhysicalVolume;
    G4VPhysicalVolume *SiPMPhysicalVolume;
    G4String name;

    // For square lattice, 0,0 is middle of 4 sipms
    for (G4double x = -maxCoord - halfPitch; x <= maxCoord - halfPitch; x += pitch_) {
        for (G4double y = -maxCoord - halfPitch; y <= maxCoord - halfPitch; y += pitch_) {

            std::vector<double> point;
            point.push_back(x);
            point.push_back(y);
            lattice_points.push_back(point); // Add the lattice point to the vector of lattice points

            //Handle x,y values to be stored and printed in format %f.2
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << x;
            std::string x_str = stream.str();
            stream.str("");
            stream << std::fixed << std::setprecision(2) << y;
            std::string y_str = stream.str();


            name = "SiPM_(x,y)=(" + x_str + "," + y_str + ")"; 
            SiPMPhysicalVolume = new G4PVPlacement(
                        0,                // no rotation
                        G4ThreeVector(x,y,zSiPM),  // at (0,0,0)
                        SiPMLogicalVolume, // its logical volume
                        name,               //its name
                        worldLogicalVolume,   //its mother volume
                        false,            // no boolean operation
                        0,// copy number, this time each SiPM has a unique copy number
                        false);          // checking overlaps



            name = "fiber_(x,y)=(" + x_str + "," + y_str + ")";
            fiberCorePhysicalVolume = new G4PVPlacement(
                                                    0,                // no rotation
                                                    G4ThreeVector(x,y,zFiber),  // at (0,0,0)
                                                    fiberCoreLogicalVolume, // its logical volume
                                                    name,               //its name
                                                    worldLogicalVolume,   //its mother volume
                                                    false,            // no boolean operation
                                                    0,                // copy number
                                                    false);        // checking overlaps                                                   



            if (claddingExists){
                name = "fiber_cladding_(x,y)=(" + x_str + "," + y_str + ")";
                fiberCladdingPhysicalVolume = new G4PVPlacement(
                                                        0,                // no rotation
                                                        G4ThreeVector(x,y,zFiber),  // at (0,0,0)
                                                        fiberCladdingLogicalVolume, // its logical volume
                                                        name,               //its name
                                                        worldLogicalVolume,   //its mother volume
                                                        false,            // no boolean operation
                                                        0,                // copy number
                                                        false);          // checking overlaps
            }



            if (wallsExists){
                name = "wall_(x,y)=(" + x_str + "," + y_str + ")";
                new G4PVPlacement(
                                0,                // no rotation
                                G4ThreeVector(x,y,zTeflonWall),  // at (0,0,0)
                                teflonWallLogicalVolume, // its logical volume
                                name,               //its name
                                worldLogicalVolume,   //its mother volume
                                false,            // no boolean operation
                                0,                // copy number
                                false);          // checking overlaps)
            }



            // name = "TPB_(x,y)=(" + x_str + "," + y_str + ")";
            fiberTPBPhysicalVolume = new G4PVPlacement(
                                                    0,                // no rotation
                                                    G4ThreeVector(x,y,zFiberTPB),  // at (0,0,0)
                                                    fiberTPBLogicalVolume, // its logical volume
                                                    "TPB_Fiber",               //its name
                                                    worldLogicalVolume,   //its mother volume
                                                    false,            // no boolean operation
                                                    0,                // copy number
                                                    false);          // checking overlaps)



            // Border surfaces
            new G4LogicalBorderSurface("TPB-Fiber",
                                        fiberTPBPhysicalVolume,
                                         fiberCorePhysicalVolume,
                                         TPBFiberSurface);


            new G4LogicalBorderSurface("Fiber-Cladding",
                            fiberCorePhysicalVolume,
                             fiberCladdingPhysicalVolume,
                             VikuitiCoating);

        }
    }



    ///// TEFLON HOLDER PLATE /////
    G4double holderInnerRadius = 0;
    G4double holderOuterRadius = barrelInnerRadius;

    // Set holder hole size in case of cladding / no cladding
    G4double holderHoleSize; 
    if (claddingExists){holderHoleSize = claddingSizeOuter;} // WITH CLADDING 
    else {holderHoleSize = fiberSize;} // WITHOUT CLADDING

    G4double holeSpacing = pitch_;
    G4int nHoles = numOfFibers;


    G4Tubs *holder = new G4Tubs("Teflon_Holder", //name
                            holderInnerRadius, //inner radius
                            holderOuterRadius, //outer radius
                            holderWidth, //length
                            0., //initial revolve angle
                            2*M_PI); //final revolve angle



    G4Box *holderHole = new G4Box("Holder_Hole",
                                holderHoleSize,
                                holderHoleSize,
                                holderWidth);


    G4double zHolder = distanceFiberHolder_;

    G4Transform3D holeTransform;
    G4double x;
    G4double y;

    G4cout << "Number of rows in lattice_points: " << lattice_points.size() << G4endl;

    G4MultiUnion* multiUnionHolder = new G4MultiUnion("MultiUnion_Holder");
    for (G4int i=0; i<nHoles*nHoles; i++) {
        x = lattice_points[i][0];
        y = lattice_points[i][1];
        holeTransform = G4Translate3D(x,y,0);
        multiUnionHolder->AddNode(*holderHole, holeTransform);
    }
    multiUnionHolder -> Voxelize();

    // Subtract the multi-union of holes from the solid
    G4SubtractionSolid* holePatternHolder = new G4SubtractionSolid("Hole_Pattern_Placement", holder, multiUnionHolder);

    G4LogicalVolume *holePatternHolderLogical = new G4LogicalVolume(holePatternHolder, Teflon, "Hole_Pattern_Placement");
    new G4LogicalSkinSurface( "Teflon_Holder", holePatternHolderLogical, teflonSurface); // makes reflections in holes
    

    G4Color yellow = G4Color::Yellow();
    G4VisAttributes* holderColor = new G4VisAttributes(yellow);
    holePatternHolderLogical->SetVisAttributes(holderColor);

    G4VPhysicalVolume *holePatternHolderPhysical;
    if (holderExists){
        holePatternHolderPhysical = new G4PVPlacement(0,
                                                        G4ThreeVector(0,0,zHolder),
                                                        holePatternHolderLogical,
                                                        "Hole_Pattern_Placement",
                                                        worldLogicalVolume,
                                                        false,
                                                        0,
                                                        false);

        G4cout << "Physical holder OK !!" << G4endl << G4endl;
    }


    // ///// TPB LAYER FOR TEFLON HOLDER /////
    G4double holderTPBCoatingInnerRadius = holderInnerRadius;
    G4double holderTPBCoatingOuterRadius = holderOuterRadius;
    G4double holderTPBCoatingHoleSize = holderHoleSize;
    G4double holderTPBCoatingWidth = TPBFiberWidth; //half width
    G4double holderTPBCoatingLocationZ = zHolder - holderWidth - holderTPBCoatingWidth;

    G4Tubs *holderTPB = new G4Tubs("TPB_Holder", //name
                            holderTPBCoatingInnerRadius, //inner radius
                            holderTPBCoatingOuterRadius, //outer radius
                            holderTPBCoatingWidth, //length
                            0., //initial revolve angle
                            2*M_PI); //final revolve angle

    G4Box *holderHoleTPB= new G4Box("TPB_Holder",
                                holderTPBCoatingHoleSize,
                                holderTPBCoatingHoleSize,
                                holderTPBCoatingWidth);


    G4MultiUnion* multiUnionTPB = new G4MultiUnion("TPB_Holder");
    for (G4int i=0; i<nHoles*nHoles; i++) {

        x = lattice_points[i][0];
        y = lattice_points[i][1];
        holeTransform = G4Translate3D(x,y,0);
        multiUnionTPB->AddNode(*holderHoleTPB, holeTransform);
    }
    multiUnionTPB -> Voxelize();

    // Subtract the multi-union of holes from the solid
    G4SubtractionSolid *holePatternTPB = new G4SubtractionSolid("TPB_Holder", holderTPB, multiUnionTPB);

    G4LogicalVolume *holePatternTPBLogical = new G4LogicalVolume(holePatternTPB, TPB, "TPB_Holder");


    holePatternTPBLogical->SetVisAttributes(colorTPB);

    G4VPhysicalVolume *holePatternTPBPhysical;
    if (holderTPBExist){
        holePatternTPBPhysical = new G4PVPlacement(0,
                                                    G4ThreeVector(0,0,holderTPBCoatingLocationZ),
                                                    holePatternTPBLogical,
                                                    "TPB_Holder",
                                                    worldLogicalVolume,
                                                    false,
                                                    0,
                                                    false);

        G4cout << "Physical holder TPB OK !!" << G4endl << G4endl;
    }



    ///// SENSITIVE DETECTORS /////

    // Set the sensitive detector for these volumes
    G4String SD_name = "SquareFiberSD";
    SquareFiberSD* squareFiberSD = new SquareFiberSD(SD_name, sipmOutputFile_, tpbOutputFile_);
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    sdManager->AddNewDetector(squareFiberSD);
    fiberTPBLogicalVolume->SetSensitiveDetector(squareFiberSD);
    holePatternTPBLogical->SetSensitiveDetector(squareFiberSD);
    SiPMLogicalVolume->SetSensitiveDetector(squareFiberSD);



 ///////////////////   THINGS TO KEEP FOR NOW    ///////////////////


    // ///// Create world center /////

    // G4Box *worldCenter = new G4Box("World_Center", edge/1000, edge/1000, edge/1000);
    // G4LogicalVolume *worldCenterLogical = new G4LogicalVolume(worldCenter,Xe, "World_Center");
    // new G4PVPlacement(
    //             0,                // no rotation
    //             G4ThreeVector(0.,0.,0.),  // at (0,0,0)
    //             worldCenterLogical, // its logical volume
    //             "World",          // its name
    //             worldLogicalVolume, // its mother  volume
    //             false,            // no boolean operation
    //             0,                // copy number
    //             true);          // checking overlaps



    // ///// HEXAGONAL LATTICE WITH CLADDING /////
    // for(G4int i=0; i<=(n-1); i++) {
    //     yFiber = (sqrt(3)*i*pitch_)/2.0;

    //     for (G4int j = 0; j < (2*n-1-i); j++) {
    //         xFiber = (-(2*n-i-2)*pitch_)/2.0 + j*pitch_;
    //         //insert core
    //         new G4PVPlacement(
    //                         0,                // no rotation
    //                         G4ThreeVector(xFiber,yFiber,zFiber),  // at (0,0,0)
    //                         fiberCoreLogicalVolume, // its logical volume
    //                         "Fiber_Core",          // its name
    //                         worldLogicalVolume, // its mother  volume
    //                         false,            // no boolean operation
    //                         count,              // copy number
    //                         true);          // checking overlaps

    //         //insert cladding
    //         new G4PVPlacement(
    //                         0,                // no rotation
    //                         G4ThreeVector(xFiber,yFiber,zFiber),  // at (0,0,0)
    //                         fiberCoatingLogicalVolume, // its logical volume
    //                         "Fiber_Cladding",               //its name
    //                         worldLogicalVolume,   //its mother volume
    //                         false,            // no boolean operation
    //                         count,          // copy number
    //                         true);          // checking overlaps

    //         count+= 1;

    //         if (yFiber!=0) {
    //             //insert core
    //             new G4PVPlacement(
    //                             0,                // no rotation
    //                             G4ThreeVector(xFiber,-yFiber,zFiber),  // at (0,0,0)
    //                             fiberCoreLogicalVolume, // its logical volume
    //                             "Fiber_Core",          // its name
    //                             worldLogicalVolume, // its mother  volume
    //                             false,            // no boolean operation
    //                             count,              // copy number
    //                             true);          // checking overlaps

    //             //insert cladding
    //             new G4PVPlacement(
    //                             0,                // no rotation
    //                             G4ThreeVector(xFiber,-yFiber,zFiber),  // at (0,0,0)
    //                             fiberCoatingLogicalVolume, // its logical volume
    //                             "Fiber_Cladding",               //its name
    //                             worldLogicalVolume,   //its mother volume
    //                             false,            // no boolean operation
    //                             count,           // copy number
    //                             true);          // checking overlaps

    //             count += 1;
    //         }
    //     }
    // }




    // ///// ROUND OPTICAL FIBER /////


    // // Tube core

    // G4double fiberSize = 1.5*mm;
    // G4double fiberLength = 0.5*cylLength - delta/2; // when sipm is inside cap

    // G4Tubs *fiberCore = new G4Tubs("Fiber_Core", //name
    //                        0, // r_inner
    //                        fiberSize, // r_outer
    //                        fiberLength, // length
    //                        0,
    //                        2*M_PI); // revolve angle

    // G4LogicalVolume *fiberCoreLogicalVolume = new G4LogicalVolume(
    //                                                     fiberCore, //the fiber object
    //                                                     //Polyethylene, //material of fiber
    //                                                     PMMA,
    //                                                     "Fiber_Core"); //name

    // G4Color white(1,1,1);
    // G4VisAttributes* roundFiberCoreColor = new G4VisAttributes(white);
    // fiberCoreLogicalVolume->SetVisAttributes(roundFiberCoreColor);
    // // new G4LogicalSkinSurface( "Fiber_Cladding", fiberCoreLogicalVolume, fiberOpticalSurface);



    // cladding for a ROUND optical fiber
   // G4OpticalSurface *cladding = new G4OpticalSurface("Cladding", //name
   //                            r2Fiber, //inner radius
   //                            r2Coating, //outer radius
   //                            tubeLength, //length
   //                            0., //initial revolve angle
   //                            2*M_PI); //finale revolve angle


   // G4LogicalVolume *coating1LogicalVolume = new G4LogicalVolume(
   //                                                     coating1, //the coating object
   //                                                     Si, //material of fiber
   //                                                     "Coating1"); //name

   //  new G4PVPlacement(
   //             0,                // no rotation
   //             G4ThreeVector(),  // at (0,0,0)
   //             coating1LogicalVolume, // its logical volume
   //             "Coating1",          // its name
   //             barrel,   // its mother  volume
   //             false,            // no boolean operation
   //             0,                // copy number
   //             true);          // checking overlaps



}//close Construct()




G4ThreeVector SquareOpticalFiber::GenerateVertex(const G4String& region) const {

    if (region == "CENTER") {
        return G4ThreeVector(0,0,0);
    }
    else if (region == "AD_HOC") {
        return specific_vertex_;
    }
    else if (region == "FACE_RANDOM_DIST"){
        x_dist_ = std::uniform_real_distribution<double>(-1.5*mm, 1.5*mm);
        y_dist_ = std::uniform_real_distribution<double>(-1.5*mm, 1.5*mm);
        G4double xGen = x_dist_(gen_);
        G4double yGen = y_dist_(gen_);
        G4double zGen = specific_vertex_.z();
        G4ThreeVector random_vertex = G4ThreeVector(xGen, yGen, zGen);
        return random_vertex;
    }
    else if (region == "LINE_SOURCE_EL"){
        G4double xGen = specific_vertex_.x();
        G4double yGen = specific_vertex_.y();
        G4double ELGapEnd = distanceFiberHolder_ - 0.5*holderThickness_ - TPBThickness_ - distanceAnodeHolder_;
        G4double ELGapStart = ELGapEnd - ELGap_;
        z_dist_ = std::uniform_real_distribution<double>(ELGapStart, ELGapEnd);
        G4double zGen = z_dist_(gen_);
        G4ThreeVector genPoint(xGen, yGen, zGen);
        return genPoint;
    }
    else {
      G4Exception("[SquareOpticalFiber]", "GenerateVertex()", FatalException,
          "Unknown vertex generation region!");
      return G4ThreeVector();
    }

} // close GenerateVertex


} // close namespace
