const modelViewerComparison1 = document.querySelector("model-viewer#modelViewerComparison1");
const modelViewerComparison2 = document.querySelector("model-viewer#modelViewerComparison2");
const modelViewerComparison3 = document.querySelector("model-viewer#modelViewerComparison3");
// // Initialize the selection panel images
// $('#comparisonSelectionPanel .selectable-image').each((i, img) => {
//     img.src = `static/comparison/${img.getAttribute('name')}/image.jpg`;
// })


// Click an image to select the case

document.querySelectorAll('#thumbnail-comparison img, #thumbnail-comparison video').forEach(el => {
    el.addEventListener('click', () => {
        const glbSrc = el.getAttribute('data-glb');
        const modelViewer = document.getElementById('modelViewerComparison1');
        modelViewer.setAttribute('src', glbSrc);
        
        // Remove border from all elements
        document.querySelectorAll('#thumbnail-comparison img, #thumbnail-comparison video').forEach(element => {
            element.style.border = 'none';
        });
        
        // Add border to clicked element
        el.style.border = '6px solid #43a3f6';

        const name = el.getAttribute('name');
        console.log('Selected thumbnail name:', name);
        const baseline = document.getElementById('comparisonBaselineSelection').value;

        // Store the selected name as an attribute on the thumbnail-comparison container
        document.getElementById('thumbnail-comparison').setAttribute('data-selected-name', name);

        const meshPath1 = `resources/comparison/ours/${name}.glb`
        const meshPath2 = `resources/comparison/dust3r/${name}.glb`
        const meshPath3 = `resources/comparison/${baseline}/${name}.glb`

        modelViewerComparison1.src = meshPath1;
        modelViewerComparison1.cameraOrbit = "180deg auto auto";
        modelViewerComparison1.resetTurntableRotation(0);
        modelViewerComparison1.jumpCameraToGoal();
        modelViewerComparison2.src = meshPath2;
        modelViewerComparison2.cameraOrbit = "180deg auto auto";
        modelViewerComparison2.resetTurntableRotation(0);
        modelViewerComparison2.jumpCameraToGoal();
        modelViewerComparison3.src = meshPath3;
        modelViewerComparison3.cameraOrbit = "180deg auto auto";
        modelViewerComparison3.resetTurntableRotation(0);
        modelViewerComparison3.jumpCameraToGoal();

    });
});

// const comparisonSelection = document.getElementById('thumbnail-comparison');
// comparisonSelection.addEventListener('click', function(event) {
//     const img = event.target.closest('.selectable-image'); 
//     if (!img || img.classList.contains('selected')) 
//         return;

//     // Load the corresponding model
//     const name = img.getAttribute('name');
//     const baseline = document.getElementById('comparisonBaselineSelection').value;

//     const meshPath1 = `static/comparison/${name}/mesh_ours.glb`, meshPath2 = `static/comparison/${name}/mesh_${baseline}.glb`;
//     const texturePath = `static/comparison/${name}/image.jpg`;
    
//     modelViewerComparison1.src = meshPath1;
//     modelViewerComparison1.texturePath = texturePath;
//     modelViewerComparison1.resetView();
//     modelViewerComparison1.showPoster();
    
//     modelViewerComparison2.src = meshPath2;
//     modelViewerComparison2.texturePath = texturePath;
//     modelViewerComparison2.resetView();
//     modelViewerComparison2.showPoster(); 
// });

// Dropdown to select the baseline method
document.getElementById('comparisonBaselineSelection').addEventListener('change', function (event) {
    // Get the name of the selected image/video from the data attribute
    const name = document.getElementById('thumbnail-comparison').getAttribute('data-selected-name');

    console.log('Selected comparison name:', name);
    
    const baseline = document.getElementById('comparisonBaselineSelection').value;
    const meshPath3 = `resources/glbs/${baseline}/${name}.glb`;
    modelViewerComparison3.src = meshPath3;

});
