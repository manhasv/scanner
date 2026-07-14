const input = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const button = document.getElementById("scanBtn");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

let selectedFile = null;
let image_id = null;
let corners = null;
let imageWidth = null;
let imageHeight = null;
let displayCorners = [];
let dragging = false;
let activeCorner = -1;

const CORNER_RADIUS = 12;

input.onchange = async () => {

    selectedFile = input.files[0];

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/preview",{
        method:"POST",
        body:formData
    });

    const data = await response.json();

    image_id = data.image_id
    corners = data.corners;
    imageHeight = data.height
    imageWidth = data.width

    preview.src = data.preview_url
};

preview.onload = () => {

    overlay.width = preview.clientWidth;
    overlay.height = preview.clientHeight;

    const scaleX = overlay.width / imageWidth;
    const scaleY = overlay.height / imageHeight

    displayCorners = corners.map(c => [
        c[0] * scaleX,
        c[1] * scaleY
    ]);

    drawCorners();
};

button.onclick = async () => {

    if (!image_id){
        alert("Choose an image first.");
        return;
    }

    const scaleX = imageWidth / overlay.width;
    const scaleY = imageHeight / overlay.height;

    corners = displayCorners.map(c => [
        Math.ceil(c[0] * scaleX),
        Math.ceil(c[1] * scaleY)
    ]);

    const response = await fetch("/scan",{
        method:"POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            image_id: image_id,
            corners: corners
        })
    });

    const blob = await response.blob();

    result.src = URL.createObjectURL(blob);
};

overlay.addEventListener("mousedown", (event) => {

    const rect = overlay.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const idx = findCorner(x, y);

    if (idx !== -1) {

        dragging = true;
        activeCorner = idx;

    }

});

overlay.addEventListener("mousemove", (event) => {

    if (!dragging)
        return;

    const rect = overlay.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    displayCorners[activeCorner] = [x, y];

    drawCorners();

});

overlay.addEventListener("mouseup", () => {

    dragging = false;
    activeCorner = -1;
    drawCorners();
});

function drawCorners() {

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    ctx.beginPath();

    ctx.moveTo(displayCorners[0][0], displayCorners[0][1]);

    for(let i = 1; i < displayCorners.length; i++){
        ctx.lineTo(displayCorners[i][0], displayCorners[i][1]);
    }

    ctx.closePath();

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;

    ctx.stroke();

    for (const corner of displayCorners){
        ctx.beginPath();
        ctx.arc(
            corner[0],
            corner[1],
            8,
            0,
            Math.PI*2
        );
        ctx.fillStyle =
            dragging && i === activeCorner
            ? "red"
            : "blue";
        ctx.fill();
    }
}

function findCorner(x, y) {

    for (let i = 0; i < displayCorners.length; i++) {

        const dx = x - displayCorners[i][0];
        const dy = y - displayCorners[i][1];

        if (Math.sqrt(dx * dx + dy * dy) < CORNER_RADIUS) {
            return i;
        }
    }

    return -1;
}