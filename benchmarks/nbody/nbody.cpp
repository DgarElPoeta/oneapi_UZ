Nbody* nbody = reinterpret_cast<Nbody*>(opts->p_problem);
uint numBodies = nbody->size;
float delT = nbody->delT;
float espSqr = nbody->espSqr;
float* currentPos = (float*)nbody->pos_in;
float* currentVel = (float*)nbody->vel_in;
float* newPos = (float*)nbody->pos_out;
float* newVel = (float*)nbody->vel_out;
//nBodyCPUReference(nbody->size, nbody->delT, nbody->espSqr, (float*)nbody->pos_in, (float*)nbody->vel_in, (float*)nbody->pos_out, (float*)nbody->vel_out);

// Iterate for all samples
for (cl_uint i = offset; i < size+offset; ++i) {
    int myIndex = 4 * i;
    float acc[3] = { 0.0f, 0.0f, 0.0f };
    for (cl_uint j = 0; j < numBodies; ++j) {
        float r[3];
        int index = 4 * j;

        float distSqr = 0.0f;
        for (int k = 0; k < 3; ++k) {
            r[k] = currentPos[index + k] - currentPos[myIndex + k];

            distSqr += r[k] * r[k];
        }

        float invDist = 1.0f / sqrt(distSqr + espSqr);
        float invDistCube = invDist * invDist * invDist;
        float s = currentPos[index + 3] * invDistCube;

        for (int k = 0; k < 3; ++k) {
            acc[k] += s * r[k];
        }
    }

    for (int k = 0; k < 3; ++k) {
        newPos[myIndex + k] =
                currentPos[myIndex + k] + currentVel[myIndex + k] * delT + 0.5f * acc[k] * delT * delT;
        newVel[myIndex + k] = currentVel[myIndex + k] + acc[k] * delT;
    }
    newPos[myIndex + 3] = currentPos[myIndex + 3];
}
