/* Borrowed from 
 * https://threejs.org/docs/#examples/controls/OrbitControls
 */
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

var scene = new THREE.Scene();

var settings = {
	displacementScale: 0.2
}

var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 10000);

var controls = new THREE.OrbitControls( camera );

/* Create a sample cube */
var geometry = new THREE.PlaneGeometry(1, 1, 300, 300);
var texture = new THREE.TextureLoader().load('unwrapped_phase.png');
var material = new THREE.MeshStandardMaterial({
	map: texture,
	displacementMap: texture,
	displacementScale: settings.displacementScale,
});

var cube = new THREE.Mesh(geometry, material);
scene.add( cube );

var light = new THREE.AmbientLight(0xffffff, 1);
scene.add( light );

camera.position.set(0, 2, 2);
controls.update();


function initGui() {
	var gui = new dat.GUI();

	gui.add(settings, "displacementScale").min(0).max(1).onChange(function(val) {
		material.displacementScale = val;
	});
}

function animate() {
	requestAnimationFrame(animate);

	controls.update();
	renderer.render(scene, camera);
}

initGui();
animate();

function onWindowResize() {

	camera.aspect = window.innerWidth / window.innerHeight;
	
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
}
window.onresize = onWindowResize;
