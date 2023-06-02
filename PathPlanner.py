import os
import unittest
import vtk, qt, ctk, slicer, random
import sitkUtils as su
import SimpleITK as sitk
from slicer.ScriptedLoadableModule import *
import logging
import numpy


#
# PathPlanner
#

class PathPlanner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PathPlanner"  # TODO make this more human readable by adding spaces
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Rachel Sparks (King's College London)"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
This file was originally developed by Rachel Sparks (King's College London) for use with the 7MRI0070 module. Aknowledgments go to Chiratchraya Akaneevanich and Tanvi Pilsokar for their contribution to this file. 
"""


#
# PathPlannerWidget
#

class PathPlannerWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # input volume selector
        #
        self.inputImageSelector = slicer.qMRMLNodeComboBox()
        self.inputImageSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.inputImageSelector.selectNodeUponCreation = True
        self.inputImageSelector.addEnabled = False
        self.inputImageSelector.removeEnabled = False
        self.inputImageSelector.noneEnabled = False
        self.inputImageSelector.showHidden = False
        self.inputImageSelector.showChildNodeTypes = False
        self.inputImageSelector.setMRMLScene( slicer.mrmlScene )
        self.inputImageSelector.setToolTip( "Pick the input image to the algorithm." )
        parametersFormLayout.addRow("Input Target Volume: ", self.inputImageSelector)

        #critical structure selector 1
        self.inputcriticalImage1Selector = slicer.qMRMLNodeComboBox()
        self.inputcriticalImage1Selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.inputcriticalImage1Selector.selectNodeUponCreation = True
        self.inputcriticalImage1Selector.addEnabled = False
        self.inputcriticalImage1Selector.removeEnabled = False
        self.inputcriticalImage1Selector.noneEnabled = False
        self.inputcriticalImage1Selector.showHidden = False
        self.inputcriticalImage1Selector.showChildNodeTypes = False
        self.inputcriticalImage1Selector.setMRMLScene(slicer.mrmlScene)
        self.inputcriticalImage1Selector.setToolTip("Pick the critical structure image to the algorithm.")
        parametersFormLayout.addRow(" Input first critical structure: ", self.inputcriticalImage1Selector)

        #critical structure selector 2
        self.criticalImage2Selector = slicer.qMRMLNodeComboBox()
        self.criticalImage2Selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.criticalImage2Selector.selectNodeUponCreation = True
        self.criticalImage2Selector.addEnabled = False
        self.criticalImage2Selector.removeEnabled = False
        self.criticalImage2Selector.noneEnabled = False
        self.criticalImage2Selector.showHidden = False
        self.criticalImage2Selector.showChildNodeTypes = False
        self.criticalImage2Selector.setMRMLScene(slicer.mrmlScene)
        self.criticalImage2Selector.setToolTip("Pick the critical structure image to the algorithm.")
        parametersFormLayout.addRow("Input second critical structure: ", self.criticalImage2Selector)

        #entry points
        self.inputFiducialSelectorEntries = slicer.qMRMLNodeComboBox()
        self.inputFiducialSelectorEntries.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.inputFiducialSelectorEntries.selectNodeUponCreation = True
        self.inputFiducialSelectorEntries.addEnabled = False
        self.inputFiducialSelectorEntries.removeEnabled = False
        self.inputFiducialSelectorEntries.noneEnabled = False
        self.inputFiducialSelectorEntries.showHidden = False
        self.inputFiducialSelectorEntries.showChildNodeTypes = False
        self.inputFiducialSelectorEntries.setMRMLScene(slicer.mrmlScene)
        self.inputFiducialSelectorEntries.setToolTip("Pick the input fiducials that you wish to be entry points.")
        parametersFormLayout.addRow("Input Entry Fiducials: ", self.inputFiducialSelectorEntries)

        #target points
        self.inputFiducialSelectorTargets = slicer.qMRMLNodeComboBox()
        self.inputFiducialSelectorTargets.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.inputFiducialSelectorTargets.selectNodeUponCreation = True
        self.inputFiducialSelectorTargets.addEnabled = False
        self.inputFiducialSelectorTargets.removeEnabled = False
        self.inputFiducialSelectorTargets.noneEnabled = False
        self.inputFiducialSelectorTargets.showHidden = False
        self.inputFiducialSelectorTargets.showChildNodeTypes = False
        self.inputFiducialSelectorTargets.setMRMLScene(slicer.mrmlScene)
        self.inputFiducialSelectorTargets.setToolTip("Pick the input fiducials that you wish to be target points.")
        parametersFormLayout.addRow("Input Target Fiducials: ", self.inputFiducialSelectorTargets)
        #
        # output fiducial selector
        #
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Pick the output fiducials to the algorithm.")
        parametersFormLayout.addRow("Output Fiducials: ", self.outputSelector)


        # retrieve max length 
        self.length = qt.QLineEdit()
        self.length.setText("50")
        self.length.setValidator(qt.QDoubleValidator(10, 100, 1))
        parametersFormLayout.addRow("Input maximum trajectory length", self.length)

        #
        # check box to trigger taking screen shots for later use in tutorials
        #
        self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
        self.enableScreenshotsFlagCheckBox.checked = 0
        self.enableScreenshotsFlagCheckBox.setToolTip(
            "If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
        parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run the algorithm."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

       

        # connections
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.inputFiducialSelectorEntries.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.inputImageSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Refresh Apply button state
        self.onSelect()

    def cleanup(self):
        pass

    def onSelect(self):
        self.applyButton.enabled = self.inputImageSelector.currentNode() and self.inputcriticalImage1Selector.currentNode() and self.criticalImage2Selector.currentNode() and self.outputSelector.currentNode() and self.inputFiducialSelectorEntries.currentNode() and self.inputFiducialSelectorTargets.currentNode()
       

    def onApplyButton(self):
        logic = PathPlannerLogic()
        enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
        logic.SetEntryPoints(self.inputFiducialSelectorEntries.currentNode())
        logic.SetTargetPoints(self.inputFiducialSelectorTargets.currentNode())
        logic.SetOutputPoints(self.outputSelector.currentNode())
        logic.SetInputImage(self.inputImageSelector.currentNode())
        logic.SetCritical1(self.inputcriticalImage1Selector.currentNode())
        logic.SetCritical2(self.criticalImage2Selector.currentNode())
        logic.MaxLength((int(self.length.text)))
        complete = logic.run(enableScreenshotsFlag)
        if not complete:
            print('I encountered an error')


#PathPlannerLogic


class PathPlannerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  #Define variables

    inside = list()      
    myTrajectories = list()  
    interpolatedTrajectories = list()
    goodTrajectories = list()
    lineDistance = list()
    goodLengthTrajectories = list() 
    distanceToCritical = list() 


    def SetEntryPoints(self, entryNode):
        self.myEntries = entryNode

    def SetTargetPoints(self, targetNode):
        self.myTargets = targetNode

    def SetInputImage(self, imageNode):
        if (self.hasImageData(imageNode)):
            self.myImage = imageNode

    def SetCritical1(self, critical1Node):
        self.Critical1 = critical1Node

    def SetCritical2(self, critical2Node):
        self.Critical2 = critical2Node

    def SetOutputPoints(self, outputNode):
        self.myOutputs = outputNode

    def MaxLength(self, length):
        self.maxlength = length


    def hasImageData(self, volumeNode):
        """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True

    def isValidInputOutputData(self, inputVolumeNode, inputTargetFiducialsNode, inputEntryFiducialsNodes,
                               outputFiducialsNode):
        """Validates if the output is not the same as input
    """
        if not inputVolumeNode:
            logging.debug('isValidInputOutputData failed: no input volume node defined')
            return False
        if not inputTargetFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input target fiducials node defined')
            return False
        if not inputEntryFiducialsNodes:
            logging.debug('isValidInputOutputData failed: no input entry fiducials node defined')
            return False
        if not outputFiducialsNode:
            logging.debug('isValidInputOutputData failed: no output fiducials node defined')
            return False
        if inputTargetFiducialsNode.GetID() == outputFiducialsNode.GetID():
            logging.debug(
                'isValidInputOutputData failed: input and output fiducial nodes are the same. Create a new output to avoid this error.')
            return False
        return True

    #compute trajectory pairs
    def ComputeTrajectories(self, myEntries, inside, myTrajectories):
        
        for x in range(0, self.myEntries.GetNumberOfFiducials()):
            for y in range(0, self.myTargets.GetNumberOfFiducials()):
                posE = [0, 0, 0]
                self.myEntries.GetNthFiducialPosition(x, posE)
                posT = [0, 0, 0]
                self.myTargets.GetNthFiducialPosition(y, posT)

                myTrajectories.append((posE, posT))
        print(myTrajectories)
        return myTrajectories


    #interpolate trajectories to get line 
    def interpolateTrajectories(self, myTrajectories , mask, interpolatedTrajectories):

        spaces= numpy.array(mask.GetSpacing())/2                                                            # calculate half spacing 
        for i in range(0, len(myTrajectories)):
            
            
            coords = list()
            pt = numpy.array(myTrajectories[i])
            dist = numpy.linalg.norm(pt[0]-pt[1])
            npoints = int(round(dist/spaces[0]))

            z = tuple(numpy.linspace(pt[0][2], pt[1][2], npoints))
            x = tuple(numpy.linspace(pt[0][0], pt[1][0], npoints))
            y = tuple(numpy.linspace(pt[0][1], pt[1][1], npoints))

            for j in range(0, npoints):
                coords.append([x[j], y[j], z[j]])
        
            interpolatedTrajectories.append(coords)

        return interpolatedTrajectories


    def run(self, enableScreenshots):
        if not self.isValidInputOutputData(self.myImage, self.myTargets, self.myEntries, self.myOutputs):
            slicer.util.errorDisplay('Input data is not valid.')
            return False

        logging.info('Processing started')

        # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
        pointPicker = PickPointsMatrix()
        pointPicker.run(self.myImage, self.myTargets, self.myOutputs)
        print('Total Target Points : ', self.myTargets.GetNumberOfFiducials())
        print('Number of Target Points within Mask : ', len(self.inside))

        # Create list of trajectory pairs
        self.ComputeTrajectories(self.myEntries, self.inside, self.myTrajectories)
        print('The possible trajectory pairs are:', self.myTrajectories)

        # Create list of interpolated trajectories
        self.interpolateTrajectories(self.myTrajectories, self.Critical1, self.interpolatedTrajectories)
        print('The trajectories after interpolation are:' , self.interpolatedTrajectories)

        # Avoiding critical structures 
        avoid = avoidcritical()
        avoid.run(self.myTrajectories, self.interpolatedTrajectories, self.Critical1)
        avoid.run(self.myTrajectories, self.interpolatedTrajectories, self.Critical2)

        # Remove trajectories longer than maxlength
        trajdistance = distance()
        trajdistance.run(self.myTrajectories, self.maxlength, self.goodLengthTrajectories)

        # Find optimal trajectory furthest from crtical structure
        away = distancetocrit()
        away.run(self.goodLengthTrajectories, self.interpolatedTrajectories, self.Critical1, self.myOutputs)
        entry = [0, 0, 0]
        self.myOutputs.GetNthFiducialPosition(0, entry)
        target = [0, 0, 0]
        self.myOutputs.GetNthFiducialPosition(1,target)
        print('The optimised trajectory is: ' + str(entry) + ',' + str(target))
        return True

       



        # Capture screenshot
        if enableScreenshots:
            self.takeScreenshot('PathPlannerTest-Start', 'MyScreenshot', -1)

        logging.info('Processing completed')

        return True


    



# using vtk to determine the point seems a bit convoluted. Lets see if there is an easier way
class PickPointsMatrix():
    def run(self, inputVolume, inputFiducials, inside):
        # So at the moment we have our boilerplate UI to take in an image and set of figudicals and output another set of fiducials
        # And are just printing something silly in our main call
        # In this first instance (related to task a) we are going to find the set of input fiducials that are within a mask of our input volume
        # First bit of clean up is to remove all points from the output-- otherwise rerunning will duplicate these
        
        # we can get a transformation from our input volume
        mat = vtk.vtkMatrix4x4();
        inputVolume.GetRASToIJKMatrix(mat)

        # set it to a transform type
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)

        for x in range(0, inputFiducials.GetNumberOfFiducials()):
            pos = [0, 0, 0]
            inputFiducials.GetNthFiducialPosition(x, pos)
            # get index from position using our transformation
            ind = transform.TransformPoint(pos)

            # get pixel using that index
            pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble(int(ind[0]), int(ind[1]), int(ind[2]), 0)  # looks like it needs 4 ints -- our x,y,z index and a component index (which is 0)
            if (pixelValue == 1):
                
                inside.append([pos[0], pos[1], pos[2]])
            return inside 

class avoidcritical():
    def run(self, myTrajectories, interpolatedTrajectories, critstructure):

        matrix = vtk.vtkMatrix4x4()
        critstructure.GetRASToIJKMatrix(matrix)

        trnaform = vtk.vtkTransform()
        transform.SetMatrix(matrix)

        removelist = list()
        for n in range(0, len(interpolatedTrajectories)):
            for p in range(0, len(interpolatedTrajectories[n])):
                position = interpolatedTrajectories[n][p]
                tposition = transform.TransformPoint(position)
                pixelValue = critstructure.GetImageData().GetScalarComponentAsDouble(int(tposition[0]), int(tposition[1]), int(tposition[2]), 0)

                if (pixelValue==1):
                    remove.append(n)
                    break
                else:
                    continue
        remove.sort(reverse=True)
        for i in remove:
            myTrajectories.pop(i)
            interpolated.pop(i)
        return myTrajectories

class distance():
  def run(self,myTrajectories,maxlength,goodLengthTrajectories):
    for i in range(0, len(myTrajectories)):
      p = numpy.array(myTrajectories[i])
      d = numpy.linalg.norm(p[0] - p[1])
      if d < maxlength:
        
        goodLengthTrajectories.append(myTrajectories[i])
    return goodLengthTrajectories

class distancetocrit():
  def ComputeDistanceImageFromLabelMap(self, inputVolume):
    sitkInput = su.PullVolumeFromSlicer(inputVolume)
    distanceFilter = sitk.DanielssonDistanceMapImageFilter()
    sitkOutput = distanceFilter.Execute(sitkInput)
    outputVolume = su.PushVolumeToSlicer(sitkOutput, None, 'distanceMap')
    return outputVolume

  def run(self, goodLengthTrajectories, interpolatedTrajectories, inputVolume, outputFiducials):

    outputVolume = self.ComputeDistanceImageFromLabelMap(inputVolume)
    mt = vtk.vtkMatrix4x4()
    outputVolume.GetRASToIJKMatrix(mt)

    transform = vtk.vtkTransform()
    transform.SetMatrix(mt)

    val = list()

    for m in range(0, len(interpolatedTrajectories)):
        tot = 0

        for n in range(0, len(interpolatedTrajectories[m])):

            p = interpolatedTrajeectories[m][n]
            dmspace = transform.TransformPoint(p)
            pixelValue = outputVolume.GetImageData().GetScalarComponentAsDouble(int(dmspace[0]), int(dmspace[1]), int(dmspace[2]),0)
            tot += pixelvalue

            val.append(tot)

    imat = val.index(max(val))
    p = goodLengthTrajectories[imat]
    outputFiducials.AddFiducial(p[0][0], p[0][1], p[0][2])
    outputFiducials.AddFiducial(p[1][0], p[1][1], p[1][2])

    return outputFiducials

class PathPlannerTest(ScriptedLoadableModuleTest):
 
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        self.test_LoadData(
            'Users\fverc\OneDrive\Desktop\Lab2and3Datasets-20230124\TestSet')  # this is a hard coded path you will need to change for your system
        self.test_PathPlanner_TestOutsidePoints()
        self.test_PathPlanner_TestInsidePoints()
        self.test_PathPlanner_TestEmptyMask()
        self.test_PathPlanner_TestEmptyPoints()
        self.setUp()  # to reclear data

    def test_LoadData(self, path):
        self.delayDisplay("Starting load data test")
        isLoaded = slicer.util.loadLabelVolume(path + '/r_hippoTest.nii.gz')
        if (not isLoaded):
            self.delayDisplay('Unable to load ' + path + '/r_hippoTest.nii.gz')

        isLoaded = slicer.util.loadMarkupsFiducialList(path + '/targets.fcsv')
        if (not isLoaded):
            self.delayDisplay('Unable to load ' + path + '/targets.fcsv')

        self.delayDisplay('Test passed! All data loaded correctly')

    def test_PathPlanner_TestOutsidePoints(self):
        """ Testing points I know are outside of the mask (first point is outside of the region entirely, second is the origin.
    """

        self.delayDisplay("Starting test points outside mask.")
        #
        # get out image node
        mask = slicer.util.getNode('r_hippoTest')

        # I am going to hard code two points -- both of which I know are not in my mask
        outsidePoints = slicer.vtkMRMLMarkupsFiducialNode()
        outsidePoints.AddFiducial(-1, -1, -1)  # this is outside of our image bounds
        cornerPoint = mask.GetImageData().GetOrigin()
        outsidePoints.AddFiducial(cornerPoint[0], cornerPoint[1], cornerPoint[2])  # we know our corner pixel is no 1

        # run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, outsidePoints, returnedPoints)

        # check if we have any returned fiducials -- this should be empty
        if (returnedPoints.GetNumberOfFiducials() > 0):
            self.delayDisplay(
                'Test failed. There are ' + str(returnedPoints.GetNumberOfFiducials()) + ' return points.')
            return

        self.delayDisplay('Test passed! No points were returned.')

    def test_PathPlanner_TestInsidePoints(self):
        """ Testing points I know are inside of the mask (first point is outside of the region entirely, second is the origin.
    """
        self.delayDisplay("Starting test points inside mask.")
        mask = slicer.util.getNode('r_hippoTest')

        # I am going to hard code one point I know is within my mask
        insidePoints = slicer.vtkMRMLMarkupsFiducialNode()
        insidePoints.AddFiducial(152.3, 124.6, 108.0)
        insidePoints.AddFiducial(145, 129, 108.0)

        # run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, insidePoints, returnedPoints)
        # check if we have any returned fiducials -- this should be 1
        if (returnedPoints.GetNumberOfFiducials() != 2):
            self.delayDisplay(
                'Test failed. There are ' + str(returnedPoints.GetNumberOfFiducials()) + ' return points.')
            return

        self.delayDisplay('Test passed!' + str(returnedPoints.GetNumberOfFiducials()) + ' points were returned.')

    def test_PathPlanner_TestEmptyMask(self):
        """Test for a null case where the mask is empty."""
        self.delayDisplay("Starting test points for empty mask.")
        mask = slicer.vtkMRMLLabelMapVolumeNode()
        mask.SetAndObserveImageData(vtk.vtkImageData())

        targets = slicer.util.getNode('targets')
        # run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, targets, returnedPoints)
        self.delayDisplay('Test passed! Empty mask dont break my code.')

    def test_PathPlanner_TestEmptyPoints(self):
        """Test for a null case where the markup fiducials is empty."""
        self.delayDisplay("Starting test points for empty points.")
        mask = slicer.util.getNode('r_hippoTest')

        # Empty point set
        insidePoints = slicer.vtkMRMLMarkupsFiducialNode()

        # run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, insidePoints, returnedPoints)
        self.delayDisplay('Test passed! Empty points dont break my code.')

    def test_PathPlanner_MaxDist(self):
        "Test if a trajectory pair exceeds the maximum distance"""
        self.delayDisplay("Starting test points exceeding the max dist.")
        filteredtraj = list()
        trajectory = [[[250,250,250],[100,100,100]],[[200,200,200],[100,100,100]]]
        finddistance().run(trajectory, 200, filteredtraj)
        #the function above should eliminate 1 and keep the other
        if (len(filteredtraj) != 1):
          self.delayDisplay('Test failed. There are ' + str(len(filteredtraj)) + ' return points.')
          return

        self.delayDisplay('Test passed! Successfully eliminated trajectories that exceed the limit')