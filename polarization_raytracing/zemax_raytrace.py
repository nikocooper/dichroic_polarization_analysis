def trace_through_zos(raysets, pth, surflist, nrays, wave, global_coords):
    """Traces initialized rays through a zemax opticstudio file
    Written by Jaren Ashcraft for Poke, a polarization raytracing package for python.

    Parameters
    ----------
    raysets : np.ndarray
        4 x Nrays array containing normalized pupil coordinates and field coordinates. Structure is like
        [x1,x2,...,xN]
        [y1,y2,...,yN]
        [l1,l2,...,lN]
        [m1,m2,...,mN]
    pth : str
        path to Zemax opticstudio file. Supports .zmx extension, .zos is untested but should work
    surflist : list of ints
        list of surface numbers to trace to and record the position of. The rays will hit every surface in the optical system,
        this just tells the Raybundle if the information at that point should be saved
    wave : int, optional
        wavelength number in ZOS file, by default 1
    global_coords : bool, optional
        whether to use global coordinates or local coordinates. Defaults to global coordinates.
        PRT uses global coordinates
        GBD uses local coordinates

    Returns
    -------
    positions : list of ndarrays
        list containing [xData,yData,zData]. Each array contains positions indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    
    directions : list
        list containing [lData,mData,nData]. Each array contains direction cosines indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    normals: list
        list containing [l2Data,m2Data,n2Data]. Each array contains surface normals indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    opd : ndarray
        Array containing the total optical path of a ray indexed by
        0 = rayset
        1 = surface
        2 = coordinate

    """
    import numpy as np
    import zosapi
    from System import Enum, Int32, Double, Array
    import clr, os

    # known directory
    # dll = os.path.join(os.path.dirname(os.path.realpath(__file__)),r'RayTrace.dll')
    dll = os.path.dirname(__file__) + r"\RayTrace.dll"
    clr.AddReference(dll)

    import BatchRayTrace

    zos = zosapi.App()
    TheSystem = zos.TheSystem
    ZOSAPI = zos.ZOSAPI
    TheSystem.LoadFile(pth, False)

    # Check to make sure the ZOSAPI is working
    if TheSystem.LDE.NumberOfSurfaces < 4:
        print("File was not loaded correctly")
        exit()

    if surflist[-1]["surf"] > TheSystem.LDE.NumberOfSurfaces:
        print("last surface > num surfaces, setting last surface to num surfaces")
        surflist[-1]["surf"] = TheSystem.LDE.NumberOfSurfaces

    maxrays = raysets[0].shape[-1]

    # Dimension 0 is ray set, Dimension 1 is surface, dimension 2 is coordinate
    # Satisfies broadcasting rules!
    xData = np.empty([len(raysets), len(surflist), maxrays])
    yData = np.empty([len(raysets), len(surflist), maxrays])
    zData = np.empty([len(raysets), len(surflist), maxrays])

    lData = np.empty([len(raysets), len(surflist), maxrays])
    mData = np.empty([len(raysets), len(surflist), maxrays])
    nData = np.empty([len(raysets), len(surflist), maxrays])

    l2Data = np.empty([len(raysets), len(surflist), maxrays])
    m2Data = np.empty([len(raysets), len(surflist), maxrays])
    n2Data = np.empty([len(raysets), len(surflist), maxrays])

    mask = np.empty([len(raysets), len(surflist), maxrays])

    # Necessary for GBD calculations, might help PRT calculations
    opd = np.empty([len(raysets), len(surflist), maxrays])

    for rayset_ind, rayset in enumerate(raysets):

        # Get the normalized coordinates
        Px = rayset[0]
        Py = rayset[1]
        Hx = rayset[2]
        Hy = rayset[3]

        for surf_ind, surfdict in enumerate(surflist):

            surf = surfdict["surf"]

            # Some ZOS-API setup
            tool = TheSystem.Tools.OpenBatchRayTrace()
            normUnpol = tool.CreateNormUnpol(maxrays, ZOSAPI.Tools.RayTrace.RaysType.Real, surf)
            reader = BatchRayTrace.ReadNormUnpolData(tool, normUnpol)
            reader.ClearData()

            # THIS OVER-INITIALIZES THE NUMBER OF RAYS.
            # This initialization is weird because it requires allocating space for a square of rays
            # so there will be extra which we remove later.
            rays = reader.InitializeOutput(nrays)

            # Add rays to reader
            reader.AddRay(wave, Hx, Hy, Px, Py, Enum.Parse(ZOSAPI.Tools.RayTrace.OPDMode, "None"))

            isfinished = False

            # Read rays
            while not isfinished:
                segments = reader.ReadNextBlock(rays)
                if segments == 0:
                    isfinished = True

            # Global Coordinate Conversion
            # Have to pre-allocate a sysDbl for this method to execute
            # TODO: This is a properly ugly line of code
            sysDbl = Double(1.0)
            # fmt: off
            (success, 
            R11, R12, R13, \
            R21, R22, R23, \
            R31, R32, R33, \
            XO, YO, ZO) = TheSystem.LDE.GetGlobalMatrix(
                int(surf), 
                sysDbl, sysDbl, sysDbl, sysDbl,
                sysDbl, sysDbl, sysDbl, sysDbl,
                sysDbl, sysDbl, sysDbl, sysDbl,
            )
            # fmt: on
            # Did the raytrace succeed?
            if success != 1:
                print("Ray Failure at surface {}".format(surf))

            # Global Rotation Matrix
            # fmt: off
            Rmat = np.array([[R11, R12, R13], 
                             [R21, R22, R23], 
                             [R31, R32, R33]])

            position = np.array(
                [np.array(list(rays.X)),
                 np.array(list(rays.Y)),
                 np.array(list(rays.Z))]
            )
            # fmt: on

            # I think this is just per-surface so it doesn't really need to be a big list, just a single surface.
            # TODO: Change later when cleaning up the code
            offset = np.zeros(position.shape)
            offset[0, :] = XO
            offset[1, :] = YO
            offset[2, :] = ZO

            # fmt: off
            angle = np.array(
                [np.array(list(rays.L)),
                 np.array(list(rays.M)),
                 np.array(list(rays.N))]
            )

            normal = np.array(
                [np.array(list(rays.l2)),
                 np.array(list(rays.m2)),
                 np.array(list(rays.n2))]
            )
            # fmt: on

            OPD = np.array(list(rays.opd))

            rays_that_passed = np.array(list(rays.vignetteCode))
            rays_that_passed = rays_that_passed[:maxrays]

            # rotate into global coordinates - necessary for PRT
            if global_coords == True:
                print("tracing with global coordinates")
                position = offset + Rmat @ position
                angle = Rmat @ angle
                normal = Rmat @ normal

            # Filter the values at the end because ZOS allocates extra space
            position = position[:, :maxrays]
            angle = angle[:, :maxrays]
            normal = normal[:, :maxrays]
            OPD = OPD[:maxrays]

            # Append data to lists along the surface dimension
            xData[rayset_ind, surf_ind] = position[0]
            yData[rayset_ind, surf_ind] = position[1]
            zData[rayset_ind, surf_ind] = position[2]

            lData[rayset_ind, surf_ind] = angle[0]
            mData[rayset_ind, surf_ind] = angle[1]
            nData[rayset_ind, surf_ind] = angle[2]

            l2Data[rayset_ind, surf_ind] = normal[0]
            m2Data[rayset_ind, surf_ind] = normal[1]
            n2Data[rayset_ind, surf_ind] = normal[2]

            # I don't think we need R and O, but might be useful to store just in case. Commenting out for now
            # R.append(Rmat)
            # O.append(offset)
            opd[rayset_ind, surf_ind] = OPD
            mask[rayset_ind, surf_ind] = rays_that_passed

            # always close your tools
            tool.Close()

    # This isn't necessary but makes the code more readable
    positions = [xData, yData, zData]
    directions = [lData, mData, nData]
    normals = [l2Data, m2Data, n2Data]

    # Just a bit of celebration
    print(
        "{nrays} Raysets traced through {nsurf} surfaces".format(
            nrays=rayset_ind + 1, nsurf=surf_ind + 1
        )
    )

    # And finally return everything
    return positions, directions, normals, opd, mask