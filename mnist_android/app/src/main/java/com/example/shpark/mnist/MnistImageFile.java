package com.example.shpark.mnist;

/**
 * Created by shpark on 2017. 4. 30..
 */

import java.io.IOException;

/**
 *
 * MNIST database image file. Contains additional header information for the
 * number of rows and columns per each entry.
 *
 */
public class MnistImageFile extends MnistDbFile {
    private int rows;
    private int cols;

    /**
     * Creates new MNIST database image file ready for reading.
     *
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws IOException
     * @throws FileNotFoundException
     */
    public MnistImageFile(String name, String mode) throws IOException {
        super(name, mode);

        // read header information
        rows = readInt();
        cols = readInt();
    }

    /**
     * Reads the image at the current position.
     *
     * @return matrix representing the image
     * @throws IOException
     */
    public int[][] readImage() throws IOException {
        int[][] dat = new int[getRows()][getCols()];
        for (int i = 0; i < getCols(); i++) {
            for (int j = 0; j < getRows(); j++) {
                dat[i][j] = readUnsignedByte();
            }
        }
        return dat;
    }

    public int[] readImage2() throws IOException {
        int[] dat = new int[getRows() * getCols()];
        for (int i = 0; i < getCols(); i++) {
            for (int j = 0; j < getRows(); j++) {
                dat[i * getCols() + j] = readUnsignedByte();
            }
        }
        return dat;
    }

    /** Read the specified number of images from the current position, to a byte[nImages][rows*cols]
     * Note that MNIST data set is stored as unsigned bytes; this method returns signed bytes without conversion
     * (i.e., same bits, but requires conversion before use)
     * @param nImages Number of images
     */
    public byte[][] readImagesUnsafe(int nImages) throws IOException{
        byte[][] out = new byte[nImages][0];
        for( int i=0; i<nImages; i++){
            out[i] = new byte[rows*cols];
            read(out[i]);
        }
        return out;
    }

    /**
     * Move the cursor to the next image.
     *
     * @throws IOException
     */
    public void nextImage() throws IOException {
        super.next();
    }

    /**
     * Move the cursor to the previous image.
     *
     * @throws IOException
     */
    public void prevImage() throws IOException {
        super.prev();
    }

    @Override
    protected int getMagicNumber() {
        return 2051;
    }

    /**
     * Number of rows per image.
     *
     * @return int
     */
    public int getRows() {
        return rows;
    }

    /**
     * Number of columns per image.
     *
     * @return int
     */
    public int getCols() {
        return cols;
    }

    @Override
    public int getEntryLength() {
        return cols * rows;
    }

    @Override
    public int getHeaderSize() {
        return super.getHeaderSize() + 8; // to more integers - rows and columns
    }
}
